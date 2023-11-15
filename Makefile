CXX=g++
CXXFLAGS= -pthread
NVCC=nvcc
BIN=bin
ERA_SRC=Eratosthenes_sieve
SUN_SRC=Sundaram_sieve

all:
#indentations below are used for proper display in the help target
	@echo "Please use 'make <target>'"
	@echo "  compile_era:				compile eratosthenes cpu"
	@echo "  compile_era_gpu:			compile eratosthenes gpu"
	@echo "  compile_sun:				compile sundaram cpu"
	@echo "  compile_sun_gpu:			compile sundaram gpu"
	@echo "  run_era:				run eratosthenes cpu"
	@echo "  run_era_gpu:				run eratosthenes gpu"
	@echo "  run_sun:				run sundaram cpu"
	@echo "  run_sun_gpu:				run sundaram gpu"
	@echo "  clean:				remove bin folder"

create:
	if [ ! -d $(BIN) ]; then mkdir $(BIN); fi

compile_era:
	$(CXX) -o $(BIN)/$(ERA_SRC)_cpu $(ERA_SRC)/CPU/main.cpp $(CXXFLAGS)

compile_era_gpu:
	$(NVCC) -o $(BIN)/$(ERA_SRC)_gpu $(ERA_SRC)/GPU/main.cu

compile_sun:
	$(CXX) -o $(BIN)/$(SUN_SRC)_cpu $(SUN_SRC)/CPU/main.cpp $(CXXFLAGS)

compile_sun_gpu:
	$(NVCC) -o $(BIN)/$(SUN_SRC)_gpu $(SUN_SRC)/GPU/main.cu

run_era: compile_era
	./$(BIN)/$(ERA_SRC)_cpu

run_era_gpu: compile_era_gpu
	./$(BIN)/$(ERA_SRC)_gpu

run_sun: compile_sun
	./$(BIN)/$(SUN_SRC)_cpu

run_sun_gpu: compile_sun_gpu
	./$(BIN)/$(SUN_SRC)_gpu

clean:
	rm $(BIN) -rf