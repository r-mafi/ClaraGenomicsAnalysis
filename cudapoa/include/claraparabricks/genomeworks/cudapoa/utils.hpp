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

#include <claraparabricks/genomeworks/cudapoa/batch.hpp>
#include <claraparabricks/genomeworks/utils/signed_integer_utils.hpp>
#include <claraparabricks/genomeworks/io/fasta_parser.hpp>

#include <string>
#include <vector>
#include <sstream>
#include <fstream>
#include <cassert>
#include <chrono>

namespace claraparabricks
{

namespace genomeworks
{

namespace cudapoa
{

/// \brief Create a small set of batch-sizes to increase GPU parallelization.
///        Input POA groups are binned based on their sizes. Similarly sized poa_groups are grouped in the same bin.
///        Separating smaller POA groups from larger ones allows to launch a larger number of groups concurrently.
///        This increase in parallelization translates to faster runtime
///        This multi-batch strategy is particularly useful when input POA groups sizes display a large variance.
///
/// \param list_of_batch_sizes [out]        a set of batch-sizes, covering all input poa_groups
/// \param list_of_groups_per_batch [out]   corresponding POA groups per batch-size bin
/// \param poa_groups [in]                  vector of input poa_groups
/// \param msa_flag [in]                    flag indicating whether MSA or consensus is going to be computed, default is consensus
/// \param band_width [in]                  band-width used in static band mode, it also defines minimum band-width in adaptive band mode
/// \param band_mode [in]                   defining which banding mod is selected: full , static or adaptive
/// \param bins_capacity [in]               pointer to vector of bins used to create separate different-sized poa_groups, if null as input, a set of default bins will be used
/// \param gpu_memory_usage_quota [in]      portion of GPU available memory that will be used for compute each cudaPOA batch, default 0.9
/// \param mismatch_score [in]              mismatch score, default -6
/// \param gap_score [in]                   gap score, default -8
/// \param match_score [in]                 match core, default 8
void get_multi_batch_sizes(std::vector<BatchConfig>& list_of_batch_sizes,
                           std::vector<std::vector<int32_t>>& list_of_groups_per_batch,
                           const std::vector<Group>& poa_groups,
                           bool msa_flag                       = false,
                           int32_t band_width                  = 256,
                           BandMode band_mode                  = BandMode::static_band,
                           std::vector<int32_t>* bins_capacity = nullptr,
                           float gpu_memory_usage_quota        = 0.9,
                           int32_t mismatch_score              = -6,
                           int32_t gap_score                   = -8,
                           int32_t match_score                 = 8);

/// \brief Resizes input windows to specified size in total_windows if total_windows >= 0
///
/// \param[out] windows      Reference to vector into which parsed window
///                          data is saved
/// \param[in] total_windows Limit windows read to total windows, or
///                          loop over existing windows to fill remaining spots.
///                          -1 ignores the total_windows arg and uses all windows in the file.
inline void resize_windows(std::vector<std::vector<std::string>>& windows, const int32_t total_windows)
{
    if (total_windows >= 0)
    {
        if (get_size(windows) > total_windows)
        {
            windows.erase(windows.begin() + total_windows, windows.end());
        }
        else if (get_size(windows) < total_windows)
        {
            int32_t windows_read = windows.size();
            while (get_size(windows) != total_windows)
            {
                windows.push_back(windows[windows.size() - windows_read]);
            }
        }

        assert(windows.size() == total_windows);
    }
}

/// \brief Parses cudapoa data file in the following format:
///        <num_sequences_in_window_0>
///        window0_seq0
///        window0_seq1
///        window0_seq2
///        ...
///        ...
///        <num_sequences_in_window_1>
///        window1_seq0
///        window1_seq1
///        window1_seq2
///        ...
///        ...
/// \param[out] windows Reference to vector into which parsed window
///                     data is saved
/// \param[in] filename Name of file with window data
/// \param[in] total_windows Limit windows read to total windows, or
///                          loop over existing windows to fill remaining spots.
///                          -1 ignored the total_windows arg and uses all windows in the file.
inline void parse_cudapoa_file(std::vector<std::vector<std::string>>& windows, const std::string& filename, int32_t total_windows)
{
    std::ifstream infile(filename);
    if (!infile.good())
    {
        throw std::runtime_error("Cannot read file " + filename);
    }
    std::string line;
    int32_t num_sequences = 0;
    while (std::getline(infile, line))
    {
        if (num_sequences == 0)
        {
            std::istringstream iss(line);
            iss >> num_sequences;
            windows.emplace_back(std::vector<std::string>());
        }
        else
        {
            windows.back().push_back(line);
            num_sequences--;
        }
    }

    resize_windows(windows, total_windows);
}

/// \brief Parses windows from 1 or more fasta files
///
/// \param[out] windows Reference to vector into which parsed window
///                     data is saved
/// \param[in] input_filepaths Reference to vector containing names of fasta files with window data
/// \param[in] total_windows Limit windows read to total windows, or
///                          loop over existing windows to fill remaining spots.
///                          -1 ignored the total_windows arg and uses all windows in the file.
inline void parse_fasta_files(std::vector<std::vector<std::string>>& windows, const std::vector<std::string>& input_paths, const int32_t total_windows, bool bonito_long_reads)
{
    const int32_t min_sequence_length = 0;
    const int32_t num_input_files     = input_paths.size();
    windows.resize(num_input_files);
    if (bonito_long_reads)
    {
        const int32_t num_input_files = 6;
        std::vector<std::shared_ptr<io::FastaParser>> fasta_parser_vec(num_input_files);
        std::vector<int> num_reads_per_file(num_input_files);
        for (int32_t i = 0; i < num_input_files; i++)
        {
            fasta_parser_vec[i]   = io::create_kseq_fasta_parser(input_paths[i], min_sequence_length, false);
            num_reads_per_file[i] = fasta_parser_vec[i]->get_num_seqences();
        }
        const int32_t num_reads = num_reads_per_file[0];

        for (int32_t i = 1; i < num_input_files; i++)
        {
            if (num_reads_per_file[i] != num_reads)
            {
                assert(false);
                std::cerr << "Failed to complete long-read sample." << std::endl;
                std::cerr << "Number of long-reads per input file do not match with each other." << std::endl;
                return;
            }
        }

        windows.resize(num_reads);
        int32_t idx = 0;
        for (auto& window : windows)
        {
            for (int32_t i = 0; i < num_input_files; i++)
            {
                window.push_back(fasta_parser_vec[i]->get_sequence_by_id(idx).seq);
            }
            idx++;
        }
    }
    else
    {
        for (int32_t i = 0; i < num_input_files; i++)
        {
            std::shared_ptr<io::FastaParser> fasta_parser = io::create_kseq_fasta_parser(input_paths[i], min_sequence_length, false);
            int32_t num_reads                             = fasta_parser->get_num_seqences();
            for (int32_t idx = 0; idx < num_reads; idx++)
            {
                windows[i].push_back(fasta_parser->get_sequence_by_id(idx).seq);
            }
        }
    }

    resize_windows(windows, total_windows);
}

/// \brief Parses golden value file with genome
///
/// \param[in] filename Name of file with reference genome
///
/// \return Genome string
inline std::string parse_golden_value_file(const std::string& filename)
{
    std::ifstream infile(filename);
    if (!infile.good())
    {
        throw std::runtime_error("Cannot read file " + filename);
    }

    std::string line;
    std::getline(infile, line);
    return line;
}

/// \brief Data structure for measuring wall-clock time spent between start and stop
struct ChronoTimer
{
private:
    /// time stamps to measure runtime
    std::chrono::time_point<std::chrono::system_clock> start_wall_clock_;
    std::chrono::time_point<std::chrono::system_clock> stop_wall_clock_;
    /// a debug flag to ensure start_timer is called before any stop_timer usage
    bool timer_initilized_ = false;

public:
    /// \brief This will record beginning of a session to be measured in the code
    void start_timer()
    {
        start_wall_clock_ = std::chrono::system_clock::now();
        timer_initilized_ = true;
    }

    /// \brief records end of a session to be measured, should be called in pair with and following start_timer().
    /// \brief The measured time (in sec) will update the record for the corresponding block among indexer, matcher or overlapper
    float stop_timer()
    {
        // start_timer was not used before calling stop_timer
        assert(timer_initilized_ == true);
        timer_initilized_ = false;
        stop_wall_clock_  = std::chrono::system_clock::now();
        auto diff         = stop_wall_clock_ - start_wall_clock_;
        auto msec         = std::chrono::duration_cast<std::chrono::milliseconds>(diff).count();
        float spent_time  = (float)msec / 1000;
        return spent_time;
    }
};

} // namespace cudapoa

} // namespace genomeworks

} // namespace claraparabricks
