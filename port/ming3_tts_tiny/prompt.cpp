#include "engine/models/ming3_tts_tiny/prompt.h"

#include <stdexcept>
#include <string>

namespace engine::models::ming3_tts_tiny {
namespace {

// Verbatim from the checkpoint's prompt_contract.py (legacy_antmmf, with_rml=true).
constexpr const char * kUserPrefix =
    "<role>HUMAN</role>Please generate speech based on the following description.\n  speaker_1:";
constexpr const char * kTextPrefix = "\n Text input:\n";
constexpr const char * kAssistantPrefix = "<role>ASSISTANT</role>";

void validate_token(const tokenizers::LlamaBpeTokenizer & tokenizer, const char * content, int32_t expected) {
    const auto found = tokenizer.find_token_id(content);
    if (!found || *found != expected) {
        throw std::runtime_error("Ming 3.0 TTS Tiny tokenizer does not match the trained token id for " +
            std::string(content));
    }
}

}  // namespace

MingTinyPromptBuilder::MingTinyPromptBuilder(tokenizers::LlamaBpeTokenizerSpec spec, MingTinySpecialTokens tokens)
    : tokenizer_(std::move(spec)), tokens_(tokens) {
    validate_token(tokenizer_, "<spk>", tokens_.speaker_start);
    validate_token(tokenizer_, "<audioPatch>", tokens_.audio_patch);
    validate_token(tokenizer_, "</spk>", tokens_.speaker_end);
    validate_token(tokenizer_, "<audio>", tokens_.audio_start);
    validate_token(tokenizer_, "<|role_end|>", tokens_.prompt_eos);
}

std::vector<std::pair<std::string, std::vector<int32_t>>> MingTinyPromptBuilder::build_segments(
    const std::string & text) const {
    std::vector<std::pair<std::string, std::vector<int32_t>>> segments;
    segments.emplace_back("user_prefix", tokenizer_.encode(kUserPrefix, true));
    segments.emplace_back("speaker_block",
        std::vector<int32_t>{tokens_.speaker_start, tokens_.audio_patch, tokens_.speaker_end});
    segments.emplace_back("text_prefix", tokenizer_.encode(kTextPrefix, true));
    segments.emplace_back("target_text", tokenizer_.encode(text, true));
    segments.emplace_back("assistant_prefix", tokenizer_.encode(kAssistantPrefix, true));
    segments.emplace_back("audio_start", std::vector<int32_t>{tokens_.audio_start});
    return segments;
}

MingTinyPromptIds MingTinyPromptBuilder::build(const std::string & text) const {
    const auto segments = build_segments(text);
    MingTinyPromptIds prompt;
    for (const auto & [name, ids] : segments) {
        if (name == "speaker_block") {
            // the <audioPatch> slot inside the speaker block
            prompt.speaker_embedding_position = prompt.ids.size() + 1;
        }
        prompt.ids.insert(prompt.ids.end(), ids.begin(), ids.end());
    }
    return prompt;
}

}  // namespace engine::models::ming3_tts_tiny
