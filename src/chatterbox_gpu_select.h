#pragma once

#include "ggml-backend.h"

#include <cctype>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

// Stage-local GPU selector for Chatterbox multi-GPU operation.
//
// selector forms:
//   "0", "1", ...        GPU ordinal among registered GPU/IGPU devices
//   "Vulkan0"            exact ggml backend device name
//   "0000:03:00.0"       exact PCI device id when available
//   "GeForce GTX 1070"   exact or unique case-insensitive description substring
//
// Returns nullptr when the selector is invalid/ambiguous or backend init fails.
// Callers should fail closed when the user explicitly requested a selector,
// rather than silently running the wrong GPU.
static inline std::string crispasr_ascii_lower(const char * s) {
    std::string out = s ? s : "";
    for (char & ch : out) {
        ch = (char) std::tolower((unsigned char) ch);
    }
    return out;
}

static inline bool crispasr_is_gpu_device(ggml_backend_dev_t dev) {
    if (!dev) {
        return false;
    }
    const enum ggml_backend_dev_type type = ggml_backend_dev_type(dev);
    return type == GGML_BACKEND_DEVICE_TYPE_GPU || type == GGML_BACKEND_DEVICE_TYPE_IGPU;
}

static inline void crispasr_log_gpu_devices(const char * stage) {
    std::fprintf(stderr, "%s: available GPU devices:\n", stage ? stage : "chatterbox");
    int ordinal = 0;
    for (size_t i = 0; i < ggml_backend_dev_count(); ++i) {
        ggml_backend_dev_t dev = ggml_backend_dev_get(i);
        if (!crispasr_is_gpu_device(dev)) {
            continue;
        }
        struct ggml_backend_dev_props props = {};
        ggml_backend_dev_get_props(dev, &props);
        std::fprintf(stderr, "  [%d] %s | %s | pci=%s\n",
                     ordinal++,
                     ggml_backend_dev_name(dev) ? ggml_backend_dev_name(dev) : "?",
                     ggml_backend_dev_description(dev) ? ggml_backend_dev_description(dev) : "?",
                     props.device_id ? props.device_id : "unknown");
    }
}

static inline ggml_backend_t crispasr_init_selected_gpu(const char * selector, int verbosity, const char * stage) {
    if (!selector || !*selector) {
        return ggml_backend_init_best();
    }

    std::vector<ggml_backend_dev_t> gpus;
    for (size_t i = 0; i < ggml_backend_dev_count(); ++i) {
        ggml_backend_dev_t dev = ggml_backend_dev_get(i);
        if (crispasr_is_gpu_device(dev)) {
            gpus.push_back(dev);
        }
    }

    if (gpus.empty()) {
        std::fprintf(stderr, "%s: requested GPU selector '%s', but no GPU devices are registered\n",
                     stage ? stage : "chatterbox", selector);
        return nullptr;
    }

    ggml_backend_dev_t chosen = nullptr;

    // Numeric selector = ordinal within GPU/IGPU devices only.
    char * end = nullptr;
    long ordinal = std::strtol(selector, &end, 10);
    if (end && *selector && *end == '\0') {
        if (ordinal >= 0 && (size_t) ordinal < gpus.size()) {
            chosen = gpus[(size_t) ordinal];
        }
    } else {
        const std::string needle = crispasr_ascii_lower(selector);
        std::vector<ggml_backend_dev_t> exact;
        std::vector<ggml_backend_dev_t> partial;

        for (ggml_backend_dev_t dev : gpus) {
            struct ggml_backend_dev_props props = {};
            ggml_backend_dev_get_props(dev, &props);

            const std::string name = crispasr_ascii_lower(ggml_backend_dev_name(dev));
            const std::string desc = crispasr_ascii_lower(ggml_backend_dev_description(dev));
            const std::string pci  = crispasr_ascii_lower(props.device_id);

            if (needle == name || (!pci.empty() && needle == pci) || needle == desc) {
                exact.push_back(dev);
            } else if (!needle.empty() && desc.find(needle) != std::string::npos) {
                partial.push_back(dev);
            }
        }

        if (exact.size() == 1) {
            chosen = exact[0];
        } else if (exact.empty() && partial.size() == 1) {
            chosen = partial[0];
        } else if (exact.size() > 1 || partial.size() > 1) {
            std::fprintf(stderr, "%s: GPU selector '%s' is ambiguous\n",
                         stage ? stage : "chatterbox", selector);
            crispasr_log_gpu_devices(stage);
            return nullptr;
        }
    }

    if (!chosen) {
        std::fprintf(stderr, "%s: GPU selector '%s' did not match any registered GPU\n",
                     stage ? stage : "chatterbox", selector);
        crispasr_log_gpu_devices(stage);
        return nullptr;
    }

    struct ggml_backend_dev_props props = {};
    ggml_backend_dev_get_props(chosen, &props);
    if (verbosity >= 1) {
        std::fprintf(stderr, "%s: selected GPU %s | %s | pci=%s\n",
                     stage ? stage : "chatterbox",
                     ggml_backend_dev_name(chosen) ? ggml_backend_dev_name(chosen) : "?",
                     ggml_backend_dev_description(chosen) ? ggml_backend_dev_description(chosen) : "?",
                     props.device_id ? props.device_id : "unknown");
    }

    return ggml_backend_dev_init(chosen, nullptr);
}
