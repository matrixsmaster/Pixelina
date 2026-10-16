VULKAN ?= 0
CXX ?= c++
GLSLC ?= glslc
PKG_CONFIG ?= pkg-config

PREFIX ?= /usr

CPPFLAGS += -Isrc -Ishaders -D_FILE_OFFSET_BITS=64 $(shell $(PKG_CONFIG) --cflags sdl2)
CXXFLAGS ?= -O3 -g
CXXFLAGS += -std=c++17 -fno-exceptions -pthread -Wall -Wextra -Wno-missing-field-initializers
LDLIBS += -pthread $(shell $(PKG_CONFIG) --libs sdl2)

CORE := os storage image json model tokenizer tensor backend_cpu kernels_cpu kernels_quant backend_vulkan convolution inference qwen_image qwen_vl qwen_vae qwen_chat
OBJECTS := $(addprefix src/,$(addsuffix .o,$(CORE)))
APP_OBJECTS := $(OBJECTS) src/ui.o src/ui_about.o src/ui_sdl.o src/ui_refdlg.o src/ui_chatdlg.o src/dlg_bookimg.o src/pixuilib.o src/main.o
HEADERS := $(wildcard src/*.h)
ASSET_HEADERS := $(patsubst assets/%.jpg,assets/%_jpg.h,$(wildcard assets/*.jpg))

ifeq ($(VULKAN),1)
CPPFLAGS += -DPIXELINA_WITH_VULKAN $(shell $(PKG_CONFIG) --cflags vulkan)
LDLIBS += $(shell $(PKG_CONFIG) --libs vulkan)
HEADERS += $(patsubst shaders/%.comp,shaders/%_shader.h,$(wildcard shaders/*.comp))
endif

.PHONY: all clean
all: pixelina

pixelina: $(APP_OBJECTS)
	$(CXX) $(LDFLAGS) $^ $(LDLIBS) -o $@

$(APP_OBJECTS): src/%.o: src/%.cpp $(HEADERS) $(ASSET_HEADERS)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) -c $< -o $@

clean:
	rm -f src/*.o shaders/*.spv shaders/*_shader.h
	rm -f assets/*.h assets/pixelina.png tools/embed tools/icon tools/font pixelina

shaders/%.spv: shaders/%.comp shaders/decode.glsl src/vulkan_ops.h src/model_types.h
	$(GLSLC) --target-env=vulkan1.0 $< -o $@

shaders/matmul_mmq.spv: shaders/matmul_mmq.comp shaders/decode.glsl src/vulkan_ops.h src/model_types.h
	$(GLSLC) --target-env=vulkan1.1 $< -o $@

shaders/attention_subgroup.spv: shaders/attention_subgroup.comp src/vulkan_ops.h src/model_types.h
	$(GLSLC) --target-env=vulkan1.1 $< -o $@

shaders/matmul_coop.spv: shaders/matmul_coop.comp shaders/decode.glsl src/vulkan_ops.h src/model_types.h
	$(GLSLC) --target-env=vulkan1.2 $< -o $@

shaders/%_shader.h: shaders/%.spv tools/embed
	tools/embed $< pixelina_$*_shader > $@

assets/%_jpg.h: assets/%.jpg tools/embed
	tools/embed $< $*_jpg > $@

tools/embed: tools/embed.cpp tools/embed.h
	$(CXX) $(CXXFLAGS) $< -o $@

tools/font: tools/font.cpp $(wildcard tools/*.h)
	$(CXX) $(CXXFLAGS) $(shell $(PKG_CONFIG) --cflags freetype2) $< $(shell $(PKG_CONFIG) --libs freetype2) -o $@

.PHONY: install uninstall
install: pixelina assets/pixelina_icon.png
	install -d $(PREFIX)/bin $(PREFIX)/share/applications $(PREFIX)/share/icons/hicolor/128x128/apps
	install -m755 pixelina $(PREFIX)/bin/pixelina
	install -m644 pixelina.desktop $(PREFIX)/share/applications/pixelina.desktop
	install -m644 assets/pixelina_icon.png $(PREFIX)/share/icons/hicolor/128x128/apps/pixelina.png
	touch $(PREFIX)/share/icons/hicolor

uninstall:
	rm -f $(PREFIX)/bin/pixelina $(PREFIX)/share/applications/pixelina.desktop $(PREFIX)/share/icons/hicolor/128x128/apps/pixelina.png
	touch $(PREFIX)/share/icons/hicolor
