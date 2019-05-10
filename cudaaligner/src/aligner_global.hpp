#pragma once

#include "cudaaligner/aligner.hpp"

namespace genomeworks {

namespace cudaaligner {

class AlignerGlobal : public Aligner
{
    public:
        AlignerGlobal(uint32_t max_query_length, uint32_t max_subject_length, uint32_t max_alignments, uint32_t device_id);
        virtual ~AlignerGlobal();
        AlignerGlobal(const AlignerGlobal&) = delete;

        virtual StatusType align_all() override;

        virtual StatusType add_alignment(const char* query, uint32_t query_length, const char* subject, uint32_t subject_length) override;

        virtual const std::vector<std::shared_ptr<Alignment>>& get_alignments() const override {
            return alignments_;
        }

        virtual uint32_t num_alignments() const {
            return alignments_.size();
        }

        virtual void set_cuda_stream(cudaStream_t stream) override {
            stream_ = stream;
        }

        virtual void reset() override;

    private:
        virtual void update_alignments_with_results();

    private:
        uint32_t max_query_length_;
        uint32_t max_subject_length_;
        uint32_t max_alignments_;
        std::vector<std::shared_ptr<Alignment>> alignments_;

        uint8_t* sequences_d_;
        uint8_t* sequences_h_;

        uint32_t* sequence_lengths_d_;
        uint32_t* sequence_lengths_h_;

        uint8_t* results_d_;
        uint8_t* results_h_;

        uint32_t* result_lengths_d_;
        uint32_t* result_lengths_h_;

        cudaStream_t stream_;

        uint32_t device_id_;
};

}

}
