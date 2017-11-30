#pragma once

#include <iostream>
#include <string>
#include <cstdint>
#include <cassert>
#include <vector>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

using namespace std;





public:

	Configuration(string boardConfiguration);
	__device__ __host__ Configuration(char** _board, lastMove _move, int numMoves, int startMoves);
	__device__ __host__ ~Configuration();

	

	__device__ __host__ friend ostream& operator<<(ostream& os, const Configuration& confg);
	__device__ __host__ bool isWinningMove();
	__device__ __host__ vector<lastMove> GenerateNextMoves(char player);
	__device__ __host__ void deleteBoard();
	__device__ __host__ char** getBoard();
	__device__ __host__ int getNMoves();
	__device__ __host__ int NumberStartMoves();
	__device__ __host__ void setNMoves(int moves);
	__forceinline__ __device__ __host__ void PrintBoard();

private:
	

	

	__device__ __host__ void SetupBoard(string boardConfiguration);
	__device__ __host__ int ValutateMove(lastMove mLastmove, int pawnInARow);
	__device__ __host__ vector<lastMove> Configuration::SortNextMoves(vector<lastMove> moves);

};