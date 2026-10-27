#include "engine/models/ming3_tts_tiny/session.h"

#include "engine/models/ming3_tts_tiny/assets.h"
#include "engine/framework/runtime/errors.h"
#include "engine/framework/runtime/session_base.h"
#include "engine/framework/runtime/spec_backed_model.h"

#include <cstdio>
#include <memory>
#include <string>

namespace engine::models::ming3_tts_tiny {
namespace {

// Milestone 1: the family exists, loads a package and reports its contract. It must
// refuse inference rather than attempt speech, so a pass/fail is unambiguous.
class MingTinySession final : public runtime::RuntimeSessionBase, public runtime::IOfflineVoiceTaskSession {
public:
    MingTinySession(runtime::TaskSpec task, const runtime::SessionOptions & options,
        std::shared_ptr<const MingTinyAssets> assets)
        : RuntimeSessionBase(options), task_(task), assets_(std::move(assets)) {}

    std::string family() const override { return "ming3_tts_tiny"; }
    runtime::VoiceTaskKind task_kind() const override { return task_.task; }
    runtime::RunMode run_mode() const override { return task_.mode; }

    void prepare(const runtime::SessionPreparationRequest &) override {
        throw runtime::InvalidRequestError(
            "Ming 3.0 TTS Tiny is loaded as a structural probe: inference is not implemented in this build");
    }

    runtime::TaskResult run(const runtime::TaskRequest &) override {
        throw runtime::InvalidRequestError(
            "Ming 3.0 TTS Tiny is a structural probe: generation is not implemented (milestone 1). "
            "The package, embedded spec, tokenizer and tensor contract were verified at load time.");
    }

private:
    runtime::TaskSpec task_;
    std::shared_ptr<const MingTinyAssets> assets_;
};

}  // namespace

std::shared_ptr<runtime::IVoiceModelLoader> make_ming3_tts_tiny_loader() {
    runtime::SpecBackedVoiceModelConfig<MingTinyAssets> config;
    config.family = "ming3_tts_tiny";
    config.load_assets = load_ming3_tts_tiny_assets;
    config.create_session = [](const runtime::TaskSpec & task, const runtime::SessionOptions & options,
        std::shared_ptr<const MingTinyAssets> assets,
        std::shared_ptr<const engine::model_spec::ModelContract> contract) {
        if (task.task != runtime::VoiceTaskKind::Tts && task.task != runtime::VoiceTaskKind::VoiceCloning &&
            task.task != runtime::VoiceTaskKind::VoiceDesign) {
            throw runtime::InvalidRequestError("Ming 3.0 TTS Tiny supports tts, clone and voice_design");
        }
        if (contract == nullptr) {
            throw runtime::InvalidRequestError("Ming 3.0 TTS Tiny requires an embedded schema v1 model contract");
        }
        std::printf("[ming3_tts_tiny] contract: schema v1 loaded for family ming3_tts_tiny, "
                    "task=%d mode=%d\n", static_cast<int>(task.task), static_cast<int>(task.mode));
        std::fflush(stdout);
        return std::make_unique<MingTinySession>(task, options, std::move(assets));
    };
    return runtime::make_spec_backed_voice_loader(std::move(config));
}

}  // namespace engine::models::ming3_tts_tiny
