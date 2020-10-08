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

#include <iostream>
#include <string>
#include <iomanip>
#include <claraparabricks/genomeworks/cudapoa/utils.hpp> // for get_multi_batch_sizes()
#include "application_parameters.hpp"
#include <spoa/spoa.hpp>
#include <omp.h>

namespace claraparabricks
{

namespace genomeworks
{

namespace cudapoa
{

std::unique_ptr<Batch> initialize_batch(int32_t mismatch_score,
                                        int32_t gap_score,
                                        int32_t match_score,
                                        bool msa,
                                        const double gpu_mem_allocation,
                                        const BatchConfig& batch_size)
{
    // Get device information.
    int32_t device_count = 0;
    GW_CU_CHECK_ERR(cudaGetDeviceCount(&device_count));
    assert(device_count > 0);

    size_t total = 0, free = 0;
    cudaSetDevice(0); // Using first GPU for sample.
    cudaMemGetInfo(&free, &total);

    // Initialize internal logging framework.
    Init();

    // Initialize CUDAPOA batch object for batched processing of POAs on the GPU.
    const int32_t device_id = 0;
    cudaStream_t stream     = 0;
    size_t mem_per_batch    = gpu_mem_allocation * free; // Using 90% of GPU available memory for CUDAPOA batch.

    std::unique_ptr<Batch> batch = create_batch(device_id,
                                                stream,
                                                mem_per_batch,
                                                msa ? OutputType::msa : OutputType::consensus,
                                                batch_size,
                                                gap_score,
                                                mismatch_score,
                                                match_score);

    return std::move(batch);
}

void process_batch(Batch* batch, bool msa_flag, bool print, std::vector<int32_t>& list_of_group_ids, int id_offset, std::vector<std::string>* batch_consensus = nullptr)
{
    batch->generate_poa();

    StatusType status = StatusType::success;
    if (msa_flag)
    {
        // Grab MSA results for all POA groups in batch.
        std::vector<std::vector<std::string>> msa; // MSA per group
        std::vector<StatusType> output_status;     // Status of MSA generation per group

        status = batch->get_msa(msa, output_status);
        if (status != StatusType::success)
        {
            std::cerr << "Could not generate MSA for batch : " << status << std::endl;
        }

        for (int32_t g = 0; g < get_size(msa); g++)
        {
            if (output_status[g] != StatusType::success)
            {
                std::cerr << "Error generating  MSA for POA group " << list_of_group_ids[g + id_offset] << ". Error type " << output_status[g] << std::endl;
            }
            else
            {
                if (print)
                {
                    for (const auto& alignment : msa[g])
                    {
                        std::cout << alignment << std::endl;
                    }
                }
            }
        }
    }
    else
    {
        // Grab consensus results for all POA groups in batch.
        std::vector<std::string> consensus;          // Consensus string for each POA group
        std::vector<std::vector<uint16_t>> coverage; // Per base coverage for each consensus
        std::vector<StatusType> output_status;       // Status of consensus generation per group

        status = batch->get_consensus(consensus, coverage, output_status);
        if (status != StatusType::success)
        {
            std::cerr << "Could not generate consensus for batch : " << status << std::endl;
        }

        for (int32_t g = 0; g < get_size(consensus); g++)
        {
            if (output_status[g] != StatusType::success)
            {
                std::cerr << "Error generating consensus for POA group " << list_of_group_ids[g + id_offset] << ". Error type " << output_status[g] << std::endl;
            }
            else
            {
                if (print)
                {
                    std::cout << consensus[g] << std::endl;
                }
            }
        }

        if (batch_consensus != nullptr)
        {
            batch_consensus->insert(batch_consensus->end(), consensus.begin(), consensus.end());
        }
    }
}

void spoa_compute(const ApplicationParameters& parameters,
                  const std::vector<std::vector<std::string>>& groups,
                  const int32_t number_of_threads,
                  bool msa_flag,
                  std::vector<std::vector<std::string>>& msa,
                  std::vector<std::string>& consensus)
{
    spoa::AlignmentType atype = spoa::AlignmentType::kNW;
    int match_score           = parameters.match_score;
    int mismatch_score        = parameters.mismatch_score;
    int gap_score             = parameters.gap_score;
    int number_of_groups      = get_size<int>(groups);

    if (msa_flag)
    {
        std::vector<std::vector<std::string>> msa_local(number_of_groups); // MSA per group

#pragma omp parallel for num_threads(number_of_threads)
        for (int g = 0; g < number_of_groups; g++)
        {
            auto alignment_engine = spoa::createAlignmentEngine(atype, match_score, mismatch_score, gap_score);
            auto graph            = spoa::createGraph();
            for (const auto& it : groups[g])
            {
                auto alignment = alignment_engine->align(it, graph);
                graph->add_alignment(alignment, it);
            }
            graph->generate_multiple_sequence_alignment(msa_local[g]);
            graph->clear();
        }

        msa.insert(msa.end(), msa_local.begin(), msa_local.end());
    }
    else
    {
        consensus.resize(number_of_groups); // Consensus string for each POA group

#pragma omp parallel for num_threads(number_of_threads)
        for (int g = 0; g < number_of_groups; g++)
        {
            auto alignment_engine = spoa::createAlignmentEngine(atype, match_score, mismatch_score, gap_score);
            auto graph            = spoa::createGraph();
            for (const auto& it : groups[g])
            {
                auto alignment = alignment_engine->align(it, graph);
                graph->add_alignment(alignment, it);
            }
            consensus[g] = graph->generate_consensus();
            graph->clear();
        }
    }
}

void run_cudapoa(const ApplicationParameters& parameters, const std::vector<Group>& poa_groups,
                 float& compute_time, std::vector<std::string>* consensus = nullptr)
{
    bool benchmark = (parameters.benchmark_mode > -1) && (consensus != nullptr);
    ChronoTimer timer;
    if (benchmark)
    {
        consensus->resize(poa_groups.size());
    }
    // analyze the POA groups and create a minimal set of batches to process them all
    std::vector<BatchConfig> list_of_batch_sizes;
    std::vector<std::vector<int32_t>> list_of_groups_per_batch;

    get_multi_batch_sizes(list_of_batch_sizes,
                          list_of_groups_per_batch,
                          poa_groups,
                          parameters.msa,
                          parameters.band_width,
                          parameters.band_mode,
                          parameters.adaptive_storage,
                          parameters.graph_length,
                          parameters.predecessor_disance,
                          nullptr,
                          parameters.gpu_mem_allocation,
                          parameters.mismatch_score,
                          parameters.gap_score,
                          parameters.match_score);

    bool print = parameters.print_output;

    std::ofstream graph_output;
    if (!parameters.graph_output_path.empty())
    {
        graph_output.open(parameters.graph_output_path);
        if (!graph_output)
        {
            std::cerr << "Error opening " << parameters.graph_output_path << " for graph output" << std::endl;
            return;
        }
    }

    int32_t group_count_offset = 0;

    for (int32_t b = 0; b < get_size(list_of_batch_sizes); b++)
    {
        auto& batch_size      = list_of_batch_sizes[b];
        auto& batch_group_ids = list_of_groups_per_batch[b];

        // for storing benchmark results
        std::vector<std::string> batch_consensus;

        // Initialize batch.
        std::unique_ptr<Batch> batch = initialize_batch(parameters.mismatch_score,
                                                        parameters.gap_score,
                                                        parameters.match_score,
                                                        parameters.msa,
                                                        parameters.gpu_mem_allocation,
                                                        batch_size);

        // Loop over all the POA groups for the current batch, add them to the batch and process them.
        int32_t group_count = 0;

        for (int32_t i = 0; i < get_size(batch_group_ids);)
        {
            const Group& group = poa_groups[batch_group_ids[i]];
            std::vector<StatusType> seq_status;
            StatusType status = batch->add_poa_group(seq_status, group);

            // NOTE: If number of batch groups smaller than batch capacity, then run POA generation
            // once last POA group is added to batch.
            if (status == StatusType::exceeded_maximum_poas || (i == get_size(batch_group_ids) - 1))
            {
                // at least one POA should have been added before processing the batch
                if (batch->get_total_poas() > 0)
                {
                    // No more POA groups can be added to batch. Now process batch
                    timer.start_timer();
                    process_batch(batch.get(), parameters.msa, print, batch_group_ids, group_count, &batch_consensus);
                    compute_time += timer.stop_timer();

                    if (graph_output.is_open())
                    {
                        if (!graph_output.good())
                        {
                            throw std::runtime_error("Error writing dot file");
                        }
                        std::vector<DirectedGraph> graph;
                        std::vector<StatusType> graph_status;
                        batch->get_graphs(graph, graph_status);
                        for (auto& g : graph)
                        {
                            graph_output << g.serialize_to_dot() << std::endl;
                        }
                    }

                    // After MSA/consensus is generated for batch, reset batch to make room for next set of POA groups.
                    batch->reset();

                    // In case that number of batch groups is more than the capacity available on GPU, the for loop breaks into smaller number of groups.
                    // if adding group i in batch->add_poa_group is not successful, it wont be processed in this iteration, therefore we print i-1
                    // to account for the fact that group i was excluded at this round.
                    if (status == StatusType::success)
                    {
                        std::cerr << "Processed groups " << group_count + group_count_offset << " - " << i + group_count_offset << " (batch " << b << ")" << std::endl;
                    }
                    else
                    {
                        std::cerr << "Processed groups " << group_count + group_count_offset << " - " << i - 1 + group_count_offset << " (batch " << b << ")" << std::endl;
                    }
                }
                else
                {
                    // the POA was too large to be added to the GPU, skip and move on
                    std::cerr << "Could not add POA group " << batch_group_ids[i] << " to batch " << b << std::endl;
                    i++;
                    if (benchmark)
                    {
                        batch_consensus.push_back("");
                    }
                }

                group_count = i;
            }

            if (status == StatusType::success)
            {
                // Check if all sequences in POA group wre added successfully.
                int32_t num_dropped_seq = 0;
                for (const auto& s : seq_status)
                {
                    if (s == StatusType::exceeded_maximum_sequence_size)
                    {
                        num_dropped_seq++;
                    }
                }

                if (num_dropped_seq > 0)
                {
                    std::cerr << "Dropping " << num_dropped_seq << " sequence(s) in POA group " << batch_group_ids[i] << " because it exceeded maximum size" << std::endl;
                }

                i++;
            }

            if (status != StatusType::exceeded_maximum_poas && status != StatusType::success)
            {
                std::cerr << "Could not add POA group " << batch_group_ids[i] << " to batch " << b << ". Error code " << status << std::endl;
                i++;
                if (benchmark)
                {
                    batch_consensus.push_back("");
                }
            }
        }

        group_count_offset += get_size(batch_group_ids);

        if (benchmark)
        {
            // add batch results to the global results vector
            for (int32_t i = 0; i < get_size<int32_t>(batch_group_ids); i++)
            {
                int id            = batch_group_ids[i];
                consensus->at(id) = batch_consensus[i];
            }
        }
    }
}

// print benchmarking report
void print_benchmark_report(const ApplicationParameters& parameters, const std::vector<Group>& poa_groups,
                            const float compute_time_a, const float compute_time_b,
                            const std::vector<std::string>& consensus_a, const std::vector<std::string>& consensus_b)
{
    int32_t number_of_groups = get_size<int32_t>(poa_groups);
    bool verbose             = !parameters.compact;

    // 0 : full, 1: static, 2: adaptive, 3: static-traceback (s-traceback)
    //=============================================================================================================
    // 00 --------------------, 01 full       <-> static, 02 full      <-> adaptive , 03 full     <-> s-traceback
    // 10 static <-> full     , 11 -------------------- , 12 static    <-> adaptive , 13 static   <-> s-traceback
    // 20 adaptive <-> full   , 21 adaptive   <-> static, ------------------------- , 23 adaptive <-> s-traceback
    // 30 s-traceback <-> full, 31 straceback <-> static, 32 straceback <-> adaptive, 33 ------------------------

    std::string method_a, method_b;
    int a    = parameters.benchmark_mode / 10;
    int b    = parameters.benchmark_mode % 10;
    method_a = a == 0 ? "full      " : a == 1 ? "static    " : a == 2 ? "adaptive  " : "traceback ";
    method_b = b == 0 ? "full      " : b == 1 ? "static    " : b == 2 ? "adaptive  " : "traceback ";

    std::cerr << "\nbenchmark summary: ";
    std::cerr << method_a << " alignment vs " << method_b << "alignment\n";
    std::cerr << "=============================================================================================================\n";
    std::cerr << "Compute time (sec):               " << method_a << std::left << std::setw(17);
    std::cerr << std::fixed << std::setprecision(2) << compute_time_a;
    std::cerr << method_b << std::left << std::setw(15) << std::fixed << std::setprecision(2) << compute_time_b;
    std::cerr << "Number of groups " << number_of_groups << std::endl;
    std::cerr << "-------------------------------------------------------------------------------------------------------------\n";

    int32_t actual_number_of_bases = 0;
    for (auto& group : poa_groups)
    {
        for (auto& seq : group)
        {
            actual_number_of_bases += seq.length;
        }
    }
    float perf_a = (float)actual_number_of_bases / compute_time_a;
    float perf_b = (float)actual_number_of_bases / compute_time_b;
    std::cerr << "Performance (bases/sec):          " << method_a << std::left << std::setw(16);
    std::cerr << std::fixed << std::setprecision(2) << std::scientific << perf_a << " ";
    std::cerr << method_b << std::left << std::setw(15) << perf_b;
    if (perf_a > perf_b)
        std::cerr << "x" << std::fixed << std::setprecision(1) << perf_a / perf_b << " faster";
    else
        std::cerr << "x" << std::fixed << std::setprecision(1) << perf_b / perf_a << " slower";
    std::cerr << "\n-------------------------------------------------------------------------------------------------------------\n";
    std::vector<int32_t> consensus_lengths_a(number_of_groups);
    std::vector<int32_t> consensus_lengths_b(number_of_groups);
    int32_t sum_consensus_length_a = 0;
    int32_t sum_consensus_length_b = 0;
    for (int i = 0; i < number_of_groups; i++)
    {
        consensus_lengths_a[i] = consensus_a[i].length();
        sum_consensus_length_a += consensus_lengths_a[i];
    }
    for (int i = 0; i < number_of_groups; i++)
    {
        consensus_lengths_b[i] = consensus_b[i].length();
        sum_consensus_length_b += consensus_lengths_b[i];
    }

    std::vector<int> min_seq_length(number_of_groups);
    std::vector<int> max_seq_length(number_of_groups);
    std::vector<int> avg_seq_length(number_of_groups);

    if (verbose)
    {
        float similarity_percentage;
        for (int i = 0; i < number_of_groups; i++)
        {
            // first find min, max and avg sequence length in the group
            int min_sz = std::numeric_limits<int>::max(), max_sz = 0, avg_sz = 0;
            const auto& group = poa_groups[i];
            for (const auto& seq : group)
            {
                min_sz = std::min(min_sz, seq.length);
                max_sz = std::max(max_sz, seq.length);
                avg_sz = avg_sz + seq.length;
            }
            min_seq_length[i] = min_sz;
            max_seq_length[i] = max_sz;
            avg_seq_length[i] = avg_sz = avg_sz / get_size<int>(group);

            std::cerr << "G " << std::left << std::setw(3) << i << " (" << std::left << std::setw(6) << min_sz << ", ";
            std::cerr << std::left << std::setw(6) << max_sz << ", " << std::left << std::setw(6) << avg_sz << std::left << std::setw(5) << ")";
            std::cerr << method_a << std::left << std::setw(17) << consensus_lengths_a[i];
            std::cerr << method_b << std::left << std::setw(15) << consensus_lengths_b[i];
            similarity_percentage = 100.0f * (1.0f - (float)abs(consensus_lengths_a[i] - consensus_lengths_b[i]) / (float)consensus_lengths_b[i]);
            std::cerr << std::left << std::setw(3) << std::fixed << std::setprecision(0) << similarity_percentage << "% similar length";
            std::cerr << std::endl;
        }
    }
    else
    {
        std::cerr << "Average consensus length:       " << method_a << std::left << std::setw(17);
        std::cerr << sum_consensus_length_a / number_of_groups;
        std::cerr << method_b << std::left << std::setw(15) << sum_consensus_length_b / number_of_groups;
        float length_similarity_percentage = 100.0f * (1.0f - (float)abs(sum_consensus_length_a - sum_consensus_length_b) / (float)sum_consensus_length_b);
        std::cerr << std::left << std::setw(3) << std::fixed << std::setprecision(0) << length_similarity_percentage << "% similar length";
        std::cerr << std::endl;
    }
    if (verbose)
    {
        //print accuracy metrics
        std::cerr << "-------------------------------------------------------------------------------------------------------------\n";

        std::vector<std::vector<std::string>> consensus_results(number_of_groups);
        for (int i = 0; i < number_of_groups; i++)
        {
            consensus_results[i].push_back(consensus_a[i]);
            consensus_results[i].push_back(consensus_b[i]);
        }

        std::vector<std::vector<std::string>> msa_for_ab;
        std::vector<std::string> dummy;

        spoa_compute(parameters, consensus_results, omp_get_num_procs(), true, msa_for_ab, dummy);

        // print comparison details between method a and b consensus per window
        for (int i = 0; i < get_size(msa_for_ab); i++)
        {
            int insert_cntr   = 0;
            int delete_cntr   = 0;
            int mismatch_cntr = 0;
            int identity_cntr = 0;

            std::cerr << "G " << std::left << std::setw(3) << i << " (" << std::left << std::setw(6) << min_seq_length[i] << ", ";
            std::cerr << std::left << std::setw(6) << max_seq_length[i] << ", " << std::left << std::setw(6) << avg_seq_length[i] << std::left << std::setw(5) << ")";

            if (msa_for_ab[i].size() == 2)
            {
                const auto& target = msa_for_ab[i][1];
                const auto& query  = msa_for_ab[i][0];
                if (target.length() == query.length())
                {
                    for (int j = 0; j < target.length(); j++)
                    {
                        if (target[j] == '-')
                            insert_cntr++;
                        else if (query[j] == '-')
                            delete_cntr++;
                        else if (target[j] != query[j])
                            mismatch_cntr++;
                        else /*target[j] == query[j]*/
                            identity_cntr++;
                    }
                    float identity_percentage = 100.0f * (float)(identity_cntr) / (float)(std::min(consensus_lengths_b[i], consensus_lengths_a[i]));

                    std::cerr << "indels  " << std::left << std::setw(4) << insert_cntr << "/" << std::left << std::setw(13) << delete_cntr;
                    std::cerr << " mismatches " << std::left << std::setw(13) << mismatch_cntr;
                    std::cerr << std::left << std::setw(3) << std::fixed << std::setprecision(0) << identity_percentage << " % identity " << std::endl;
                }
                else
                {
                    std::cerr << "indels  " << std::left << std::setw(18) << "--------";
                    std::cerr << " mismatches " << std::left << std::setw(13) << "---";
                    std::cerr << std::left << std::setw(3) << std::fixed << std::setprecision(0) << "NA"
                              << " % identity " << std::endl;
                }
            }
            else
            {
                std::cerr << "indels  " << std::left << std::setw(18) << "--------";
                std::cerr << " mismatches " << std::left << std::setw(13) << "---";
                std::cerr << std::left << std::setw(3) << std::fixed << std::setprecision(0) << "NA"
                          << " % identity " << std::endl;
            }
        }
    }
    std::cerr << "=============================================================================================================\n\n";
}

int main(int argc, char* argv[])
{
    // Parse input parameters
    const ApplicationParameters parameters(argc, argv);

    // Load input data. Each window is represented as a vector of strings. The sample
    // data has many such windows to process, hence the data is loaded into a vector
    // of vector of strings.
    std::vector<std::vector<std::string>> windows;
    if (parameters.all_fasta)
    {
        parse_fasta_files(windows, parameters.input_paths, parameters.max_groups);
    }
    else
    {
        parse_cudapoa_file(windows, parameters.input_paths[0], parameters.max_groups);
    }

    // processing only a single window defined by option -D
    if (parameters.single_window > -1 && parameters.single_window < get_size<int>(windows))
    {
        auto window = windows[parameters.single_window];
        windows.resize(1);
        windows[0] = window;
    }

    // Create a vector of POA groups based on windows
    std::vector<Group> poa_groups(windows.size());
    for (int32_t i = 0; i < get_size(windows); ++i)
    {
        Group& group = poa_groups[i];
        // Create a new entry for each sequence and add to the group.
        int32_t num_reads = 0;
        for (const auto& seq : windows[i])
        {
            Entry poa_entry{};
            poa_entry.seq     = seq.c_str();
            poa_entry.length  = seq.length();
            poa_entry.weights = nullptr;
            group.push_back(poa_entry);
            num_reads++;
            if (num_reads == parameters.max_reads)
            {
                break;
            }
        }
    }

    if (parameters.sort_reads || parameters.filter_outliers)
    {
        int32_t group_id = 0;
        for (auto& g : poa_groups)
        {
            std::sort(g.begin(), g.end(), [](const Entry& s1, const Entry s2) -> bool { return s1.length > s2.length; });
            if (parameters.filter_outliers)
            {
                // use simple 1.5xIQR rule to find and filter out outlier data
                int32_t median_length = 0;
                int32_t group_size    = get_size(g);
                int32_t num_erased    = 0;
                if (group_size > 0)
                {
                    median_length = g[group_size / 2].length;
                }
                int32_t min_length = median_length * (1.0f - parameters.filter_tolerance);
                int32_t max_length = median_length * (1.0f + parameters.filter_tolerance);
                for (auto it = g.begin(); it != g.end();)
                {
                    if (it->length < min_length || it->length > max_length)
                    {
                        it = g.erase(it);
                        num_erased++;
                    }
                    else
                    {
                        it++;
                    }
                }
                if (num_erased > 0)
                {
                    std::cerr << "Removed " << num_erased << " sequence(s) in POA group " << group_id << " as outliers." << std::endl;
                }
            }
            group_id++;
        }
    }

    // print-out reads in fasta format
    if (parameters.output_fasta)
    {
        int64_t id = 0;
        for (auto& group : poa_groups)
        {
            for (auto& s : group)
            {
                std::cout << ">s" << id << std::endl;
                std::cout << s.seq << std::endl;
                id++;
            }
        }
        return 0;
    }

    // for benchmarking
    float time_a = 0.f;
    float time_b = 0.f;
    std::vector<std::string> consensus_a;
    std::vector<std::string> consensus_b;

    if (parameters.benchmark_mode == -1)
    {
        run_cudapoa(parameters, poa_groups, time_a);
    }
    else
    {
        ApplicationParameters parameters_a = parameters;
        ApplicationParameters parameters_b = parameters;

        // 0 : full, 1: static, 2: adaptive, 3: static-traceback (s-traceback)
        //=============================================================================================================
        int a = parameters.benchmark_mode / 10;
        int b = parameters.benchmark_mode % 10;
        if (a == b || a > 4 || b > 4)
        {
            std::cerr << "for benchmark (opetion -B), two different methods should be compared against each other.\n";
            std::cerr << "00       invalid       , 01    full <-> static   , 02     full <-> adaptive  , 03    full <-> s-traceback  \n";
            std::cerr << "10   static <-> full   , 11        invalid       , 12   static <-> adaptive  , 13  static <-> s-traceback  \n";
            std::cerr << "20  adaptive <-> full  , 21  adaptive <-> static ,          invalid          , 23 adaptive <-> s-traceback \n";
            std::cerr << "30 s-traceback <-> full, 31 straceback <-> static, 32 straceback <-> adaptive, 33          invalid         \n";
            return -1;
        }
        parameters_a.band_mode = a == 0 ? full_band : a == 1 ? static_band : a == 2 ? adaptive_band : a == 3 ? static_band_traceback : adaptive_band_traceback;
        parameters_b.band_mode = b == 0 ? full_band : b == 1 ? static_band : b == 2 ? adaptive_band : b == 3 ? static_band_traceback : adaptive_band_traceback;

        run_cudapoa(parameters_a, poa_groups, time_a, &consensus_a);
        run_cudapoa(parameters_b, poa_groups, time_b, &consensus_b);
    }

    if (parameters.benchmark_mode > -1)
    {
        print_benchmark_report(parameters, poa_groups, time_a, time_b, consensus_a, consensus_b);
    }

    return 0;
}

} // namespace cudapoa

} // namespace genomeworks

} // namespace claraparabricks

/// \brief main function
/// main function cannot be in a namespace so using this function to call actual main function
int main(int argc, char* argv[])
{
    return claraparabricks::genomeworks::cudapoa::main(argc, argv);
}
