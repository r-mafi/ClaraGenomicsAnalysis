/*
* Copyright (c) 2019, NVIDIA CORPORATION.  All rights reserved.
*
* NVIDIA CORPORATION and its licensors retain all intellectual property
* and proprietary rights in and to this software, related documentation
* and any modifications thereto.  Any use, reproduction, disclosure or
* distribution of this software and related documentation without an express
* license agreement from NVIDIA CORPORATION is strictly prohibited.
*/

#include <cudamapper/cudamapper.hpp>
#include <claragenomics/logging/logging.hpp>

namespace claragenomics {
    namespace cudamapper {

        StatusType Init()
        {
            if (logging::LoggingStatus::success != logging::Init())
                return StatusType::generic_error;
                
            return StatusType::success;
        }

    };
};
