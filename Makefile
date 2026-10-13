VULKAN ?= 0
CXX ?= c++
GLSLC ?= glslc
PKG_CONFIG ?= pkg-config

CPPFLAGS += -Isrc -Ishaders -D_FILE_OFFSET_BITS=64 $(shell $(PKG_CONFIG) --cflags sdl2)
CXXFLAGS ?= -O3 -g
CXXFLAGS += -std=c++17 -fno-exceptions -pthread -Wall -Wextra -Wno-missing-field-initializers
LDLIBS += -pthread $(shell $(PKG_CONFIG) --libs sdl2)

CORE := storage image json model tokenizer tensor backend_cpu kernels_cpu kernels_quant backend_vulkan convolution inference
OBJECTS := $(addprefix src/,$(addsuffix .o,$(CORE)))
APP_OBJECTS := $(OBJECTS) src/ui.o src/pixuilib.o src/main.o
HEADERS := $(wildcard src/*.h)

ifeq ($(VULKAN),1)
CPPFLAGS += -DPIXELINA_WITH_VULKAN $(shell $(PKG_CONFIG) --cflags vulkan)
LDLIBS += $(shell $(PKG_CONFIG) --libs vulkan)
HEADERS += $(patsubst shaders/%.comp,shaders/%_shader.h,$(wildcard shaders/*.comp))
endif

.PHONY: all clean
all: pixelina

pixelina: $(APP_OBJECTS)
	$(CXX) $(LDFLAGS) $^ $(LDLIBS) -o $@

$(APP_OBJECTS): src/%.o: src/%.cpp $(HEADERS) assets/pixelina_icon.h
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) -c $< -o $@

clean:
	rm -f src/*.o shaders/*.spv shaders/*_shader.h assets/pixelina_icon.h assets/pixelina.png tools/embed tools/icon pixelina

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

assets/pixelina_icon.h: assets/pixelina.jpg tools/embed
	tools/embed $< pixelina_icon > $@

tools/embed: tools/embed.cpp tools/embed.h
	$(CXX) $(CXXFLAGS) $< -o $@

tools/icon: tools/icon.cpp $(wildcard tools/*.h third_party/*.h)
	$(CXX) $(CXXFLAGS) $< -o $@

assets/pixelina.png: assets/pixelina.jpg tools/icon
	tools/icon $< $@

PREFIX ?= /usr/local
.PHONY: install uninstall
install: pixelina assets/pixelina.png
	install -d $(DESTDIR)$(PREFIX)/bin $(DESTDIR)$(PREFIX)/share/applications $(DESTDIR)$(PREFIX)/share/icons/hicolor/256x256/apps
	install -m755 pixelina $(DESTDIR)$(PREFIX)/bin/pixelina
	install -m644 pixelina.desktop $(DESTDIR)$(PREFIX)/share/applications/pixelina.desktop
	install -m644 assets/pixelina.png $(DESTDIR)$(PREFIX)/share/icons/hicolor/256x256/apps/pixelina.png

uninstall:
	rm -f $(DESTDIR)$(PREFIX)/bin/pixelina $(DESTDIR)$(PREFIX)/share/applications/pixelina.desktop $(DESTDIR)$(PREFIX)/share/icons/hicolor/256x256/apps/pixelina.png
