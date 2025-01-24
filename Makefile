BIN_DIR = bin
build: make_dir
	nvcc -O3 -o $(BIN_DIR)/electrostatic_field electrostatic_field.cu -lSDL2
make_dir:
	mkdir -p $(BIN_DIR)
run: build
	$(BIN_DIR)/electrostatic_field
clean:
	rm -rf $(BIN_DIR)

.PHONY: build run clean