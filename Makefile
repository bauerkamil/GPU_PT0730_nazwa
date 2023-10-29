CXX=g++
BIN=bin
ERA_SRC=Eratoshenes_sieve
SUN_SRC=Sundaram_sieve

all:
	@echo "Please use 'make <target>'"
	@echo "  compile1:		compile eratoshenes cpu"
	@echo "  compile2:		compile eratoshenes gpu"
	@echo "  compile3:		compile sundaram cpu"
	@echo "  compile4:		compile sundaram gpu"
	@echo "  run1:			run eratoshenes cpu"
	@echo "  run2:			run eratoshenes gpu"
	@echo "  run3:			run sundaram cpu"
	@echo "  run4:			run sundaram gpu"
	@echo "  clean:		remove bin folder"

create:
	if [ ! -d $(BIN) ]; then mkdir $(BIN); fi

compile1: create
	$(CXX) -o $(BIN)/$(ERA_SRC)_cpu $(ERA_SRC)/CPU/main.cpp

compile2: create
	$(CXX) -o $(BIN)/$(ERA_SRC)_gpu $(ERA_SRC)/GPU/main.cpp

compile3: create
	$(CXX) -o $(BIN)/$(SUN_SRC)_cpu $(SUN_SRC)/CPU/main.cpp

compile4: create
	$(CXX) -o $(BIN)/$(SUN_SRC)_gpu $(SUN_SRC)/GPU/main.cpp

run1: compile1
	./$(BIN)/$(ERA_SRC)_cpu

run2: compile2
	./$(BIN)/$(ERA_SRC)_gpu

run3: compile3
	./$(BIN)/$(SUN_SRC)_cpu

run4: compile4
	./$(BIN)/$(SUN_SRC)_gpu

clean:
	rm $(BIN) -rf