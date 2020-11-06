/*
* Copyright 2019-2020 NVIDIA CORPORATION.
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
*     http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
*/

#pragma once

#include "cudapoa_structs.cuh"

#include <claraparabricks/genomeworks/utils/cudautils.hpp>
#include <claraparabricks/genomeworks/utils/limits.cuh>

#include <stdio.h>

namespace claraparabricks
{

namespace genomeworks
{

namespace cudapoa
{

template <typename ScoreT>
__device__ __forceinline__ ScoreT* get_score_ptr_tb(ScoreT* scores, int32_t score_row, int32_t column, int32_t band_start, int32_t band_width)
{
    column = column == -1 ? 0 : column - band_start;

    int64_t score_index = static_cast<int64_t>(column) +
                          static_cast<int64_t>(score_row) * static_cast<int64_t>(band_width + CUDAPOA_BANDED_MATRIX_RIGHT_PADDING);

    return &scores[score_index];
}

template <typename ScoreT>
__device__ __forceinline__ void set_score_tb(ScoreT* scores,
                                             int32_t row,
                                             int32_t column,
                                             int32_t score_matrix_height,
                                             int32_t value,
                                             int32_t band_start,
                                             int32_t band_width)
{
    int32_t col_idx;
    if (column == -1)
    {
        col_idx = band_start;
    }
    else
    {
        col_idx = column - band_start;
    }
    // in NW with traceback buffer, score matrix is stored partially, hence row is mapped to [0, score_matrix_height) span
    row                 = row % score_matrix_height;
    int64_t score_index = static_cast<int64_t>(col_idx) +
                          static_cast<int64_t>(row) * static_cast<int64_t>(band_width + CUDAPOA_BANDED_MATRIX_RIGHT_PADDING);
    scores[score_index] = value;
}

__device__ __forceinline__ int32_t get_band_start_for_row_tb(int32_t row, float gradient, int32_t band_width, int32_t band_shift, int32_t max_column)
{
    int32_t diagonal_index = int32_t(row * gradient);
    int32_t start_pos      = max(0, diagonal_index - band_shift);
    if (max_column < start_pos + band_width)
    {
        start_pos = max(0, max_column - band_width + CELLS_PER_THREAD);
    }
    start_pos = start_pos - (start_pos % CELLS_PER_THREAD);

    return start_pos;
}

template <typename ScoreT>
__device__ __forceinline__ void initialize_band_tb(ScoreT* scores,
                                                   int32_t row,
                                                   int32_t score_matrix_height,
                                                   int32_t min_score_value,
                                                   int32_t band_start,
                                                   int32_t band_width,
                                                   int32_t lane_idx)
{
    int32_t band_end = band_start + band_width;

    band_start = max(1, band_start);

    set_score_tb(scores, row, band_start, score_matrix_height, min_score_value, band_start, band_width);

    // note: as long as CUDAPOA_BANDED_MATRIX_RIGHT_PADDING < WARP_SIZE, no need for a for loop
    for (int32_t j = lane_idx + band_end; j < band_end + CUDAPOA_BANDED_MATRIX_RIGHT_PADDING; j += WARP_SIZE)
    {
        set_score_tb(scores, row, j, score_matrix_height, min_score_value, band_start, band_width);
    }
}

template <typename TraceT>
__device__ __forceinline__ TraceT get_trace(TraceT* traceback, int32_t row, int32_t column, int32_t band_start, int32_t band_width)
{
    int64_t trace_index = static_cast<int64_t>(column - band_start) +
                          static_cast<int64_t>(row) * static_cast<int64_t>(band_width + CUDAPOA_BANDED_MATRIX_RIGHT_PADDING);
    return traceback[trace_index];
}

template <typename ScoreT>
__device__ __forceinline__ ScoreT get_score_tb(ScoreT* scores,
                                               int32_t row,
                                               int32_t column,
                                               int32_t score_matrix_height,
                                               int32_t band_width,
                                               int32_t band_shift,
                                               float gradient,
                                               int32_t max_column,
                                               const ScoreT min_score_value)
{
    int32_t band_start = get_band_start_for_row_tb(row, gradient, band_width, band_shift, max_column);
    int32_t band_end   = band_start + band_width;
    band_end           = min(band_end, max_column);

    if (((column > band_end) || (column < band_start)) && column != -1)
    {
        return min_score_value;
    }
    else
    {
        // row is mapped to [0, score_matrix_height) span
        return *get_score_ptr_tb(scores, row % score_matrix_height, column, band_start, band_width);
    }
}

template <typename SeqT, typename ScoreT, typename TraceT>
__device__ __forceinline__ void get_scores_tb(ScoreT* scores,
                                              ScoreT* scores_sh,
                                              int32_t pred_node,
                                              int32_t current_node,
                                              int32_t column,
                                              int32_t score_matrix_height,
                                              int32_t band_width,
                                              int32_t band_shift,
                                              float gradient,
                                              int32_t max_column,
                                              int32_t gap_score,
                                              int32_t match_score,
                                              int32_t mismatch_score,
                                              SeqT4<SeqT> read4,
                                              SeqT graph_base,
                                              ScoreT4<ScoreT>& score,
                                              TraceT4<TraceT>& trace)
{
    // The load instructions typically load data in 4B or 8B chunks.
    // If data is 16b (2B), then a 4B load chunk is loaded into register
    // and the necessary bits are extracted before returning. This wastes cycles
    // as each read of 16b issues a separate load command.
    // Instead it is better to load a 4B or 8B chunk into a register
    // using a single load inst, and then extracting necessary part of
    // of the data using bit arithmetic. Also reduces register count.

    int32_t band_start = get_band_start_for_row_tb(pred_node, gradient, band_width, band_shift, max_column);

    // subtract by CELLS_PER_THREAD to ensure score4_next is not pointing out of the corresponding band bounds
    int32_t band_end = band_start + band_width - CELLS_PER_THREAD;
    band_end         = min(band_end, max_column);

    if ((column > band_end || column < band_start) && column != -1)
    {
        return;
    }
    else
    {
        ScoreT4<ScoreT> score4, score4_next;
        if (pred_node + 1 == current_node)
        {
            score4.s0      = scores_sh[column - band_start];
            score4.s1      = scores_sh[column - band_start + 1];
            score4.s2      = scores_sh[column - band_start + 2];
            score4.s3      = scores_sh[column - band_start + 3];
            score4_next.s0 = scores_sh[column - band_start + 4];
        }
        else
        {
            // row is mapped to [0, score_matrix_height) span
            ScoreT4<ScoreT>* pred_scores = (ScoreT4<ScoreT>*)get_score_ptr_tb(scores, pred_node % score_matrix_height, column, band_start, band_width);
            // loads 8/16 consecutive bytes (4 ScoreTs)
            score4 = pred_scores[0];
            // need to load the next chunk of memory as well
            score4_next = pred_scores[1];
        }

        int32_t char_profile = (graph_base == read4.r0 ? match_score : mismatch_score);

        // if trace is diogonal, its value is positive and if vertical, negative
        // update score.s0, trace.t0 ----------
        if ((score4.s0 + char_profile) >= (score4.s1 + gap_score))
        {
            if ((score4.s0 + char_profile) > score.s0)
            {
                score.s0 = score4.s0 + char_profile;
                trace.t0 = current_node - pred_node;
            }
        }
        else
        {
            if ((score4.s1 + gap_score) > score.s0)
            {
                score.s0 = score4.s1 + gap_score;
                trace.t0 = -(current_node - pred_node);
            }
        }
        // update score.s1, trace.t1 ----------
        char_profile = (graph_base == read4.r1 ? match_score : mismatch_score);
        if ((score4.s1 + char_profile) >= (score4.s2 + gap_score))
        {
            if ((score4.s1 + char_profile) > score.s1)
            {
                score.s1 = score4.s1 + char_profile;
                trace.t1 = current_node - pred_node;
            }
        }
        else
        {
            if ((score4.s2 + gap_score) > score.s1)
            {
                score.s1 = score4.s2 + gap_score;
                trace.t1 = -(current_node - pred_node);
            }
        }
        // update score.s2, trace.t2  ----------
        char_profile = (graph_base == read4.r2 ? match_score : mismatch_score);
        if ((score4.s2 + char_profile) >= (score4.s3 + gap_score))
        {
            if ((score4.s2 + char_profile) > score.s2)
            {
                score.s2 = score4.s2 + char_profile;
                trace.t2 = current_node - pred_node;
            }
        }
        else
        {
            if ((score4.s3 + gap_score) > score.s2)
            {
                score.s2 = score4.s3 + gap_score;
                trace.t2 = -(current_node - pred_node);
            }
        }
        // update score.s3, trace.t3 ----------
        char_profile = (graph_base == read4.r3 ? match_score : mismatch_score);
        if ((score4.s3 + char_profile) >= (score4_next.s0 + gap_score))
        {
            if ((score4.s3 + char_profile) > score.s3)
            {
                score.s3 = score4.s3 + char_profile;
                trace.t3 = current_node - pred_node;
            }
        }
        else
        {
            if ((score4_next.s0 + gap_score) > score.s3)
            {
                score.s3 = score4_next.s0 + gap_score;
                trace.t3 = -(current_node - pred_node);
            }
        }
    }
}

template <typename SeqT,
          typename ScoreT,
          typename SizeT,
          typename TraceT,
          bool ADAPTIVE = true>
__device__ __forceinline__
    int32_t
    runNeedlemanWunschBandedTraceback(SeqT* nodes,
                                      SizeT* graph,
                                      SizeT* node_id_to_pos,
                                      int32_t graph_count,
                                      uint16_t* incoming_edge_count,
                                      SizeT* incoming_edges,
                                      uint16_t* outgoing_edge_count,
                                      SeqT* read,
                                      int32_t read_length,
                                      ScoreT* scores,
                                      TraceT* traceback,
                                      float max_buffer_size,
                                      SizeT* alignment_graph,
                                      SizeT* alignment_read,
                                      int32_t band_width,
                                      int32_t score_matrix_height,
                                      int32_t gap_score,
                                      int32_t mismatch_score,
                                      int32_t match_score,
                                      int32_t rerun)
{
    const ScoreT min_score_value = numeric_limits<ScoreT>::min() / 2;

    int32_t lane_idx = threadIdx.x % WARP_SIZE;

    //Calculate aspect ratio for the scores matrix
    float gradient = float(read_length + 1) / float(graph_count + 1);

    int32_t max_column = read_length + 1;

    // one row buffer for keeping scores of the previous node
    __shared__ ScoreT scores_sh[2][264+1];

    // Set band-width based on scores matrix aspect ratio
    //---------------------------------------------------------
    // band_shift defines distance of band_start from the scores matrix diagonal, ad-hoc rule 4
    int32_t band_shift = band_width / 2;

    if (ADAPTIVE)
    {
        // rerun code is defined in backtracking loop from previous alignment try
        // SHIFT_ADAPTIVE_BAND_TO_LEFT means traceback path was too close to the left bound of band
        // SHIFT_ADAPTIVE_BAND_TO_RIGHT means traceback path was too close to the right bound of band
        // Therefore we rerun alignment of the same read, but this time with double band-width and band_shift further to
        // the left for rerun == SHIFT_ADAPTIVE_BAND_TO_LEFT, and further to the right for rerun == SHIFT_ADAPTIVE_BAND_TO_RIGHT.
        if (rerun == SHIFT_ADAPTIVE_BAND_TO_LEFT && band_width <= CUDAPOA_MAX_ADAPTIVE_BAND_WIDTH / 2)
        {
            // ad-hoc rule 5
            band_width *= 2;
            band_shift *= 2.5;
        }
        if (rerun == SHIFT_ADAPTIVE_BAND_TO_RIGHT && band_width <= CUDAPOA_MAX_ADAPTIVE_BAND_WIDTH / 2)
        {
            // ad-hoc rule 6
            band_width *= 2;
            band_shift *= 1.5;
        }
        // check required memory and return error if exceeding max_buffer_size
        // using float to avoid 64-bit
        float required_buffer_size = static_cast<float>(graph_count) * static_cast<float>(band_width + CUDAPOA_BANDED_MATRIX_RIGHT_PADDING);
        if (required_buffer_size > max_buffer_size)
        {
            return CUDAPOA_KERNEL_NW_ADAPTIVE_STORAGE_FAILED;
        }
    }
    //---------------------------------------------------------

    // Initialise the horizontal boundary of the score matrix, initialising of the vertical boundary is done within the main for loop
    for (int32_t j = lane_idx; j < band_width + CUDAPOA_BANDED_MATRIX_RIGHT_PADDING; j += WARP_SIZE)
    {
        set_score_tb(scores, 0, j, score_matrix_height, j * gap_score, 0, band_width);
        scores_sh[0][j] = j * gap_score;
    }

    if (lane_idx == 0)
    {
#ifdef NW_VERBOSE_PRINT
        printf("graph %d, read %d\n", graph_count, read_length);
#endif
    }

    __syncwarp();

    // compute vertical and diagonal values in parallel.
    for (int32_t graph_pos = 0; graph_pos < graph_count; graph_pos++)
    {
        int32_t node_id      = graph[graph_pos];
        int32_t score_gIdx   = graph_pos + 1;
        int32_t band_start   = get_band_start_for_row_tb(score_gIdx, gradient, band_width, band_shift, max_column);
        int32_t pred_node_id = incoming_edges[node_id * CUDAPOA_MAX_NODE_EDGES];

        initialize_band_tb(scores, score_gIdx, score_matrix_height, min_score_value, band_start, band_width, lane_idx);

        int32_t first_element_prev_score = 0;
        int32_t first_element_prev_trace = 0;
        uint16_t pred_count              = 0;
        int32_t pred_idx                 = 0;

        if (lane_idx == 0)
        {
            // Initialise the vertical boundary of the score matrix
            int32_t penalty;
            pred_count      = incoming_edge_count[node_id];
            scores_sh[1][0] = scores_sh[1][band_width] = min_score_value;
            if (pred_count == 0)
            {
                first_element_prev_score = gap_score;
                first_element_prev_trace = -score_gIdx;
                // row is mapped to [0, score_matrix_height) span
                int64_t index    = static_cast<int64_t>(score_gIdx % score_matrix_height) * static_cast<int64_t>(band_width + CUDAPOA_BANDED_MATRIX_RIGHT_PADDING);
                scores[index]    = first_element_prev_score;
                scores_sh[1][0]  = first_element_prev_score;
                index            = static_cast<int64_t>(score_gIdx) * static_cast<int64_t>(band_width + CUDAPOA_BANDED_MATRIX_RIGHT_PADDING);
                traceback[index] = first_element_prev_trace;
            }
            else
            {
                int64_t index = static_cast<int64_t>(score_gIdx) * static_cast<int64_t>(band_width + CUDAPOA_BANDED_MATRIX_RIGHT_PADDING);
                pred_idx      = node_id_to_pos[pred_node_id] + 1;
                // only predecessors that are less than score_matrix_height distant can be taken into account
                if ((graph_pos - pred_idx) < score_matrix_height)
                {
                    first_element_prev_trace = -(score_gIdx - pred_idx);
                    if (band_start > CELLS_PER_THREAD && pred_count == 1)
                    {
                        first_element_prev_score = min_score_value + gap_score;
                        // fill in first column of traceback buffer
                        traceback[index] = first_element_prev_trace;
                    }
                    else
                    {
                        penalty = max(min_score_value,
                                      get_score_tb(scores, pred_idx, -1, score_matrix_height, band_width, band_shift, gradient, max_column, min_score_value));
                        // if pred_num > 1 keep checking to find max score as penalty
                        for (int32_t p = 1; p < pred_count; p++)
                        {
                            pred_node_id         = incoming_edges[node_id * CUDAPOA_MAX_NODE_EDGES + p];
                            int32_t pred_idx_tmp = node_id_to_pos[pred_node_id] + 1;
                            // only predecessors that are less than score_matrix_height distant can be taken into account
                            if ((score_gIdx - pred_idx_tmp) < score_matrix_height)
                            {
                                int32_t trace_tmp = -(score_gIdx - pred_idx_tmp);
                                int32_t score_tmp = get_score_tb(scores, pred_idx_tmp, -1, score_matrix_height, band_width, band_shift,
                                                                 gradient, max_column, min_score_value);
                                if (penalty < score_tmp)
                                {
                                    penalty                  = score_tmp;
                                    first_element_prev_trace = trace_tmp;
                                }
                            }
                        }
                        first_element_prev_score = penalty + gap_score;
                        set_score_tb(scores, score_gIdx, -1, score_matrix_height, first_element_prev_score, band_start, band_width);
                        scores_sh[1][0]  = first_element_prev_score;
                        traceback[index] = first_element_prev_trace;
                    }
                }
                else
                {
                    penalty = min_score_value;
                    // look for a predecessor which is within score_matrix_height limit
                    for (int32_t p = 1; p < pred_count; p++)
                    {
                        pred_node_id         = incoming_edges[node_id * CUDAPOA_MAX_NODE_EDGES + p];
                        int32_t pred_idx_tmp = node_id_to_pos[pred_node_id] + 1;
                        // only predecessors that are less than score_matrix_height distant can be taken into account
                        if ((score_gIdx - pred_idx_tmp) < score_matrix_height)
                        {
                            int32_t trace_tmp = -(score_gIdx - pred_idx_tmp);
                            int32_t score_tmp = get_score_tb(scores, pred_idx_tmp, -1, score_matrix_height, band_width, band_shift, gradient, max_column, min_score_value);
                            if (penalty < score_tmp)
                            {
                                penalty                  = score_tmp;
                                first_element_prev_trace = trace_tmp;
                            }
                        }
                    }
                    first_element_prev_score = penalty + gap_score;
                    set_score_tb(scores, score_gIdx, -1, score_matrix_height, first_element_prev_score, band_start, band_width);
                    scores_sh[1][0]  = first_element_prev_score;
                    traceback[index] = first_element_prev_trace;
                }
            }
        }
        pred_count = __shfl_sync(FULL_MASK, pred_count, 0);
        pred_idx   = __shfl_sync(FULL_MASK, pred_idx, 0);
        //-------------------------------------------------------------

        SeqT graph_base = nodes[node_id];

        for (int32_t read_pos = lane_idx * CELLS_PER_THREAD + band_start; read_pos < band_start + band_width; read_pos += CUDAPOA_MIN_BAND_WIDTH)
        {
            SeqT4<SeqT>* d_read4 = (SeqT4<SeqT>*)read;
            SeqT4<SeqT> read4    = d_read4[read_pos / CELLS_PER_THREAD];

            TraceT4<TraceT> trace;
            ScoreT4<ScoreT> score = {min_score_value, min_score_value, min_score_value, min_score_value};
            // note that whenever accessing a score matrix row, the row needs to be mapped to [0, score_matrix_height)
            get_scores_tb(scores, scores_sh[0], pred_idx, score_gIdx, read_pos, score_matrix_height, band_width, band_shift, gradient,
                          max_column, gap_score, match_score, mismatch_score, read4, graph_base, score, trace);

            // Perform same score updates as above, but for rest of predecessors.
            for (int32_t p = 1; p < pred_count; p++)
            {
                int32_t pred_idx_tmp = node_id_to_pos[incoming_edges[node_id * CUDAPOA_MAX_NODE_EDGES + p]] + 1;
                if ((score_gIdx - pred_idx_tmp) < score_matrix_height)
                {
                    get_scores_tb(scores, scores_sh[0], pred_idx_tmp, score_gIdx, read_pos, score_matrix_height, band_width, band_shift, gradient,
                                  max_column, gap_score, match_score, mismatch_score, read4, graph_base, score, trace);
                }
            }

            // While there are changes to the horizontal score values, keep updating the matrix.
            // So loop will only run the number of time there are corrections in the matrix.
            // The any_sync warp primitive lets us easily check if any of the threads had an update.
            bool loop = true;

            while (__any_sync(FULL_MASK, loop))
            {
                loop = false;
                // Note: computation of s3 depends on s2, s2 depends on s1 and s1 on s0.
                // If we reverse the order of computation in this loop from s3 to s0, it will increase
                // ILP. However, in longer reads where indels are more frequent, this reverse computations
                // results in larger number of iterations. Since if s0 is changed, value of s1, s2 and s3 which
                // already have been computed in parallel need to be updated again.

                // The shfl_up lets us grab a value from the lane below.
                int32_t last_score = __shfl_up_sync(FULL_MASK, score.s3, 1);
                if (lane_idx == 0)
                {
                    last_score = first_element_prev_score;
                }

                if (score.s0 < last_score + gap_score)
                {
                    score.s0 = last_score + gap_score;
                    trace.t0 = 0;
                }

                if (score.s1 < score.s0 + gap_score)
                {
                    score.s1 = score.s0 + gap_score;
                    trace.t1 = 0;
                }

                if (score.s2 < score.s1 + gap_score)
                {
                    score.s2 = score.s1 + gap_score;
                    trace.t2 = 0;
                }

                int32_t tscore = max(score.s2 + gap_score, score.s3);
                if (tscore > score.s3)
                {
                    score.s3 = tscore;
                    trace.t3 = 0;
                    loop     = true;
                }
            }

            // copy values in single row shared mem cache
            scores_sh[1][read_pos + 1 - band_start] = score.s0;
            scores_sh[1][read_pos + 2 - band_start] = score.s1;
            scores_sh[1][read_pos + 3 - band_start] = score.s2;
            scores_sh[1][read_pos + 4 - band_start] = score.s3;

            // shuffle registers to allow coalesced storing
            ScoreT s3_copy = score.s3;
            score.s3       = score.s2;
            score.s2       = score.s1;
            score.s1       = score.s0;
            score.s0       = __shfl_up_sync(FULL_MASK, s3_copy, 1);

            TraceT t3_copy = trace.t3;
            trace.t3       = trace.t2;
            trace.t2       = trace.t1;
            trace.t1       = trace.t0;
            trace.t0       = __shfl_up_sync(FULL_MASK, t3_copy, 1);

            if (lane_idx == 0)
            {
                score.s0 = first_element_prev_score;
                trace.t0 = first_element_prev_trace;
            }

            // perform coalesced write
            ScoreT4<ScoreT>* scores4_ptr = reinterpret_cast<ScoreT4<ScoreT>*>(scores);
            int64_t index                = static_cast<int64_t>((read_pos - band_start) / CELLS_PER_THREAD + static_cast<float>(score_gIdx % score_matrix_height) * static_cast<float>(band_width / CELLS_PER_THREAD + 2));
            scores4_ptr[index]           = score;
            if (lane_idx == (WARP_SIZE - 1))
            {
                scores4_ptr[index + 1] = ScoreT4<ScoreT>{s3_copy, min_score_value, min_score_value, min_score_value};
            }

            TraceT4<TraceT>* traces4_ptr = reinterpret_cast<TraceT4<TraceT>*>(traceback);
            index                        = static_cast<int64_t>((read_pos - band_start) / CELLS_PER_THREAD + static_cast<float>(score_gIdx) * static_cast<float>(band_width / CELLS_PER_THREAD + 2));
            traces4_ptr[index]           = trace;
            if (lane_idx == (WARP_SIZE - 1))
            {
                traces4_ptr[index + 1] = TraceT4<TraceT>{t3_copy, 0, 0, 0};
            }

            // Copy over the last element score and trace of the last lane into a register of first lane
            // which can be used to compute the first cell of the next warp.
            first_element_prev_score = __shfl_sync(FULL_MASK, s3_copy, WARP_SIZE - 1);
            first_element_prev_trace = __shfl_sync(FULL_MASK, t3_copy, WARP_SIZE - 1);

            __syncwarp();
        }

        // temp : copy shared mem row 1 to 0
        for (int32_t j = lane_idx; j < band_width; j += WARP_SIZE)
        {
            scores_sh[0][j] = scores_sh[1][j];
        }
        __syncwarp();
    }

    int32_t aligned_nodes = 0;
    if (lane_idx == 0)
    {
        // Find location of the maximum score in the matrix.
        int32_t i      = 0;
        int32_t j      = read_length;
        int32_t mscore = min_score_value;

        for (int32_t idx = 1; idx <= graph_count; idx++)
        {
            if (outgoing_edge_count[graph[idx - 1]] == 0)
            {
                if ((graph_count - idx) < score_matrix_height)
                {
                    int32_t s = get_score_tb(scores, idx, j, score_matrix_height, band_width, band_shift, gradient, max_column, min_score_value);
                    if (mscore < s)
                    {
                        mscore = s;
                        i      = idx;
                    }
                }
            }
        }

        // if i was not set, throw an error indicating selected score_matrix_height (i.e. max predecessor distance) is too small
        if (i == 0)
        {
            j             = 0;
            aligned_nodes = CUDAPOA_KERNEL_NW_TRACEBACK_BUFFER_FAILED;
        }

        //------------------------------------------------------------------------

        // Fill in traceback
        int32_t loop_count = 0;
        while (!(i == 0 && j == 0) && loop_count < static_cast<int32_t>(read_length + graph_count + 2))
        {
            loop_count++;

            int32_t band_start = get_band_start_for_row_tb(i, gradient, band_width, band_shift, max_column);
            TraceT trace       = get_trace(traceback, i, j, band_start, band_width);

            if (trace == 0)
            {
                // horizontal path (indel)
                alignment_graph[aligned_nodes] = -1;
                alignment_read[aligned_nodes]  = j - 1;
                j--;
            }
            else if (trace < 0)
            {
                // vertical path (indel)
                alignment_graph[aligned_nodes] = graph[i - 1];
                alignment_read[aligned_nodes]  = -1;
                i += trace;
            }
            else
            {
                // diagonal path (match/mismatch)
                alignment_graph[aligned_nodes] = graph[i - 1];
                alignment_read[aligned_nodes]  = j - 1;
                i -= trace;
                j--;

                if (ADAPTIVE)
                {
                    // no need to request rerun if (a) it's not the first run, (b) band_width == CUDAPOA_MAX_ADAPTIVE_BAND_WIDTH already
                    if (rerun == 0 && band_width < CUDAPOA_MAX_ADAPTIVE_BAND_WIDTH)
                    {
                        // check if traceback gets too close or hits the band limits, if so stop and rerun with extended band-width
                        // threshold for proximity to band limits works better if defined proportionate to the sequence length
                        int32_t threshold = max(1, max_column / 1024); // ad-hoc rule 7
                        if (j > threshold && j < max_column - threshold)
                        {
                            int32_t band_start = get_band_start_for_row(i, gradient, band_width, band_shift, max_column);
                            if (j <= band_start + threshold) // ad-hoc rule 8-a, too close to left bound
                            {
                                aligned_nodes = SHIFT_ADAPTIVE_BAND_TO_LEFT;
                                break;
                            }
                            if (j >= (band_start + band_width - threshold)) // ad-hoc rule 8-b, too close to right bound
                            {
                                aligned_nodes = SHIFT_ADAPTIVE_BAND_TO_RIGHT;
                                break;
                            }
                        }
                    }
                }
            }

            aligned_nodes++;
        }

        if (loop_count >= (read_length + graph_count + 2))
        {
            aligned_nodes = CUDAPOA_KERNEL_NW_BACKTRACKING_LOOP_FAILED;
        }

#ifdef NW_VERBOSE_PRINT
        printf("aligned nodes %d, loop count %d\n", aligned_nodes, loop_count);
#endif
    }
    aligned_nodes = __shfl_sync(FULL_MASK, aligned_nodes, 0);
    return aligned_nodes;
}

} // namespace cudapoa

} // namespace genomeworks

} // namespace claraparabricks
