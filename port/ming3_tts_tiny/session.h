#pragma once

#include "engine/framework/runtime/model.h"

namespace engine::models::ming3_tts_tiny {

std::shared_ptr<runtime::IVoiceModelLoader> make_ming3_tts_tiny_loader();

}  // namespace engine::models::ming3_tts_tiny
