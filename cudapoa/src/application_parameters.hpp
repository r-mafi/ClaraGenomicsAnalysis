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
#include <vector>
#include <string>

namespace claraparabricks
{

namespace genomeworks
{

namespace cudapoa
{

/// @brief application parameteres, default or passed through command line
class ApplicationParameters
{
public:
    /// @brief constructor reads input from command line
    /// @param argc
    /// @param argv
    ApplicationParameters(int argc, char* argv[]);

    std::vector<std::string> input_paths;
    std::string graph_output_path;
    bool all_fasta            = true;
    bool msa                  = false; // consensus by default
    bool banded               = true;
    bool adaptive             = false;
    int32_t band_width        = 256; // Band width for banded mode
    int32_t max_groups        = -1;  // -1 => infinite
    int32_t mismatch_score    = -6;
    int32_t gap_score         = -8;
    int32_t match_score       = 8;
    double gpu_mem_allocation = 0.9;
    // for benchmark mode
    bool compact          = true;
    bool output_fasta     = false;
    int32_t single_window = -1;
    int32_t max_reads     = -1;

private:
    /// \brief verifies input file formats
    /// \param input_paths input files to verify
    void verify_input_files(std::vector<std::string>& input_paths);

    /// \brief prints cudamapper's version
    /// \param exit_on_completion
    void print_version(bool exit_on_completion = true);

    /// \brief prints help message
    /// \param exit_code
    [[noreturn]] void help(int32_t exit_code = 0);
};

} // namespace cudapoa

} // namespace genomeworks

} // namespace claraparabricks
