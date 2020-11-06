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

#include <claraparabricks/genomeworks/io/fasta_parser.hpp>

#include <string>
#include <vector>

namespace claraparabricks
{

namespace genomeworks
{

namespace io
{

class FastaParserKseqpp : public FastaParser
{
public:
    /// \brief Constructor
    /// \param fasta_file Path to FASTA(.gz) file. If .gz, it must be zipped with bgzip.
    /// \param min_sequence_length Minimum length a sequence needs to be to be parsed. Shorter sequences are ignored.
    /// \param shuffle Enables shuffling reads
    FastaParserKseqpp(const std::string& fasta_file,
                      number_of_basepairs_t min_sequence_length,
                      bool shuffle);

    /// \brief Constructor used to wrap around std::vector<std::string> data
    /// \param reads raw reads data
    FastaParserKseqpp(const std::vector<std::string>& reads, const std::string& name);

    /// \brief Return number of sequences in FASTA file
    /// \return Sequence count in file
    number_of_reads_t get_num_seqences() const override;

    /// \brief Fetch an entry from the FASTA file by index position in file.
    /// \param sequence_id Position of sequence in file. If sequence_id is invalid an error is thrown.
    /// \return A reference to FastaSequence describing the entry.
    const FastaSequence& get_sequence_by_id(read_id_t sequence_id) const override;

private:
    /// All the reads from the FASTA file are stored in host RAM
    /// given a sufficiently-large FASTA file, there may not be enough host RAM
    /// on the system
    std::vector<FastaSequence> reads_;
};

} // namespace io

} // namespace genomeworks

} // namespace claraparabricks
