BIN_DIR = bin
make_dir:
	mkdir -p $(BIN_DIR)
build: make_dir
	nvcc -O3 -o $(BIN_DIR)/electrostatic_field electrostatic_field.cu -lSDL2
run: build
	$(BIN_DIR)/electrostatic_field
clean:
	rm -rf $(BIN_DIR)
