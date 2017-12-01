#pragma once

#include <stdio.h>
#include <iostream>
#include <fstream>
#include <string>
#include <limits>
#include <cstdint>
#include <cassert>
#include <vector>
#include <algorithm>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "device_functions.h"
#include "sm_32_atomic_functions.h"

using namespace std;

__device__ int gAlpha=-100;
__device__ int gBeta=100;
__device__ int gScore=0;

struct lastMove {
	int row; int column; char player; int value;
	lastMove(int _row, int _column, char _player, int _value) {
		row = _row;
		column = _column;
		player = _player;
		value = _value;
	}
};

class Configuration {
public:
	static const int ROWS = 6;  // width of the board
	static const int COLUMNS = 7; // height of the board
	static const int MIN_SCORE = -(ROWS*COLUMNS) / 2 + 3;
	static const int MAX_SCORE = (ROWS*COLUMNS + 1) / 2 - 3;
	
	int numberOfConfigurations=0;
	char* board;
	char* dev_board;
	lastMove mLastmove = lastMove(-1, -1, 'n', 0);
private:
	int NumberOfMoves;
	
	int NumberOfStartMoves;

	

public:
	Configuration() {}
	Configuration(string boardConfiguration) {
		SetupBoard(boardConfiguration);

		cudaMalloc((void **)&dev_board, 6 * 7 * sizeof(char*));
		cudaMemcpy(dev_board, board, 6 * 7 * sizeof(char*), cudaMemcpyHostToDevice);

		/*for (int i = 0; i < 6; i++) {
			cudaMalloc(&board[i], 7 * sizeof(char));
			//cudaMemcpy(dev_board[i], board[i], 7 * sizeof(char), cudaMemcpyHostToDevice);
		}
		cudaMemcpy(dev_board, board[0], 7*6 * sizeof(char), cudaMemcpyHostToDevice);
		*/
		mLastmove = lastMove(-1, -1, '0', 0);
		for each (char c in boardConfiguration)
		{
			if (c == 'X' || c == '0')
				NumberOfMoves++;
		}
		NumberOfStartMoves = NumberOfMoves;
	}

	__device__ __host__ Configuration(char* _board, lastMove _move, int numMoves, int startMoves,int numConfig) {
		_board[_move.row*COLUMNS+_move.column] = _move.player;
		mLastmove.row = _move.row;
		mLastmove.column = _move.column;
		mLastmove.player = _move.player;
		NumberOfStartMoves = startMoves;
		NumberOfMoves = numMoves + 1;
		numberOfConfigurations = numConfig;

		//cudaMalloc((void **)&dev_board, 6 * 7 * sizeof(char*));
		cudaMemcpy(dev_board, _board, 6 * 7 * sizeof(char*), cudaMemcpyDeviceToDevice);

		/*dev_board = new char[ROWS*COLUMNS];
		for (int i = 0; i < ROWS*COLUMNS; i++) {
			board[i] = _board[i];
		}*/
	}



	__device__ __host__ bool isWinningMove() {
		//cout << mLastmove.row << "   " << mLastmove.column << endl;
		if (mLastmove.row == -1)
			return false;
		int counter = 0;
		//check the column
		for (int j = 0; j < COLUMNS; j++) {
			if (dev_board[mLastmove.row*COLUMNS + j] == mLastmove.player) {
				counter++;
				if (counter >= 4)
					return true;
			}
			else
				counter = 0;
		}
		counter = 0;
		//check the row
		for (int i = 0; i < ROWS; i++) {
			if (dev_board[i*COLUMNS + mLastmove.column] == mLastmove.player) {
				counter++;
				if (counter >= 4)
					return true;
			}
			else
				counter = 0;
		}
		counter = 0;
		//check right diagonal
		for (int k = 0; (k + mLastmove.row < ROWS && k + mLastmove.column < COLUMNS); k++) {
			if (dev_board[(mLastmove.row + k)*COLUMNS + (mLastmove.column + k)] == mLastmove.player) {
				counter++;
				if (counter >= 4)
					return true;
			}
			else
				counter = 0;
		}
		counter = 0;
		for (int k = 0; (mLastmove.row - k >= 0 && mLastmove.column - k >= 0); k++) {
			if (dev_board[(mLastmove.row - k)*COLUMNS + (mLastmove.column - k)] == mLastmove.player) {
				counter++;
				if (counter >= 4)
					return true;
			}
			else
				counter = 0;
		}
		//check left diagonal
		counter = 0;
		for (int k = 0; (k + mLastmove.row < ROWS && mLastmove.column - k >= 0); k++) {
			if (dev_board[(mLastmove.row + k)*COLUMNS + (mLastmove.column - k)] == mLastmove.player) {
				counter++;

				if (counter >= 4)
					return true;
			}
			else
				counter = 0;
		}
		counter = 0;
		for (int k = 0; (mLastmove.row - k >= 0 && k + mLastmove.column < COLUMNS); k++) {
			if (dev_board[(mLastmove.row - k)*COLUMNS + (mLastmove.column + k)] == mLastmove.player) {
				counter++;

				if (counter >= 4)
					return true;
			}
			else
				counter = 0;
		}
		return false;
	}

	__device__ __host__ void PrintBoard() {
		for (int i = 0; i < Configuration::ROWS; i++) {
			for (int j = 0; j < Configuration::COLUMNS; j++) {
				printf("%c", dev_board[i*Configuration::COLUMNS+j]);
			}
			printf("\n");
		}
	}

	__device__ __host__ char* getBoard() {
		char * _board = new char[ROWS*COLUMNS];

		for (int i = 0; i < ROWS*COLUMNS; i++) {
			_board[i] = dev_board[i];
		}
		return _board;
	}

	__device__ __host__ int getNMoves() {
		return NumberOfMoves;
	}

	__device__ __host__ void setNMoves(int moves) {
		NumberOfMoves = moves;
	}

	__device__ __host__ int NumberStartMoves()
	{
		return NumberOfStartMoves;
	}


	__device__ __host__ void deleteBoard()
	{
		delete[] board;
		delete[] dev_board;
	}

	__device__ __host__ ~Configuration() {
	}

	__device__ __host__ void SetupBoard(string boardConfiguration) {
		board = new char[ROWS*COLUMNS];
		for (int i = 0; i <boardConfiguration.length(); i++) {
			board[i] = boardConfiguration[i];
		}
	}

	__device__ __host__ int ValutateMove(lastMove mLastmove, int pawnInARow) {
		int value = 0;
		if (mLastmove.row == -1)
			return value;

		int counter = 0;
		//check the column
		for (int j = (mLastmove.column - pawnInARow > 0 ? mLastmove.column - pawnInARow : 0); j < (mLastmove.column + pawnInARow < COLUMNS ? mLastmove.column + pawnInARow : COLUMNS); j++) {
			if (dev_board[mLastmove.row*COLUMNS + j] == mLastmove.player) {
				counter++;
				if (counter >= pawnInARow) {
					value += pawnInARow;
					break;
				}
			}
			else
				counter = 0;
		}
		counter = 0;
		//check the row
		for (int i = (mLastmove.row - pawnInARow > 0 ? mLastmove.row - pawnInARow : 0); i < (mLastmove.row + pawnInARow < ROWS ? mLastmove.row + pawnInARow : ROWS); i++) {
			if (dev_board[i*COLUMNS + mLastmove.column] == mLastmove.player) {
				counter++;
				if (counter >= pawnInARow) {
					value += pawnInARow;
					break;
				}
			}
			else
				counter = 0;
		}
		counter = 0;
		//check right diagonal
		for (int k = 0; (k + mLastmove.row < ROWS && k + mLastmove.column < COLUMNS); k++) {
			if (dev_board[(mLastmove.row + k)*COLUMNS + (mLastmove.column + k)] == mLastmove.player) {
				counter++;
				if (counter >= pawnInARow) {
					value += pawnInARow;
					break;
				}
			}
			else
				counter = 0;
		}
		counter = 0;
		for (int k = 0; (mLastmove.row - k >= 0 && mLastmove.column - k >= 0); k++) {
			if (dev_board[(mLastmove.row - k)*COLUMNS + (mLastmove.column - k)] == mLastmove.player) {
				counter++;
				if (counter >= pawnInARow) {
					value += pawnInARow;
					break;
				}
			}
			else
				counter = 0;
		}
		//check left diagonal
		counter = 0;
		for (int k = 0; (k + mLastmove.row < ROWS && mLastmove.column - k >= 0); k++) {
			if (dev_board[(mLastmove.row + k)*COLUMNS + (mLastmove.column - k)] == mLastmove.player) {
				counter++;
				if (counter >= pawnInARow) {
					value += pawnInARow;
					break;
				}
			}
			else
				counter = 0;
		}
		counter = 0;
		for (int k = 0; (mLastmove.row - k >= 0 && k + mLastmove.column < COLUMNS); k++) {
			if (dev_board[(mLastmove.row - k)*COLUMNS + (mLastmove.column + k)] == mLastmove.player) {
				counter++;
				if (counter >= pawnInARow) {
					value += pawnInARow;
					break;
				}
			}
			else
				counter = 0;
		}
		
		for (int i = 0; i < ROWS; i++) {
			counter = 0;
			for (int j = 0; j < COLUMNS; j++) {
				if (dev_board[i*COLUMNS + j] != mLastmove.player &&board[i*COLUMNS + j] != '-') {
					counter++;
					if (counter >= pawnInARow) {
						if (j - counter >= 1) {
							if (dev_board[i*COLUMNS + (j - counter)] == '-')
								value -= pawnInARow;
						}
						break;
					}
				}
				else
					counter = 0;
			}
		}

		for (int j = 0; j < COLUMNS; j++) {
			counter = 0;
			for (int i = 0; i < ROWS; i++) {
				if (dev_board[i*COLUMNS + j] != mLastmove.player && dev_board[i*COLUMNS + j] != '-') {
					counter++;
					if (counter >= pawnInARow) {
						if (i - counter >= 1) {
							if (dev_board[(i - counter)*COLUMNS + j] == '-')
								value -= pawnInARow;
						}
						if (i < ROWS - 1) {
							if (dev_board[(i + 1)*COLUMNS + j] == '-')
								value -= pawnInARow;
						}
						break;
					}
				}
				else
					counter = 0;
			}
		}
		return value;
	}

	//genera le sette mosse successive
	__device__ __host__ Configuration* GenerateNextMoves(char player) {

		int idx = 0;
		int counter = 0;
		for (int j = 0; j < COLUMNS; j++) {
			for (int i = ROWS - 1; i >= 0; i--) {
				idx = i*COLUMNS + j;
				if (dev_board[idx] == '-') {
					counter++;
					break;
				}
			}
		}
		numberOfConfigurations = counter;
		if (numberOfConfigurations > 0) {
			Configuration* configurations = new Configuration[counter];
			idx = 0;
			counter = 0;
			for (int j = 0; j < COLUMNS; j++) {
				for (int i = ROWS - 1; i >= 0; i--) {
					idx = i*COLUMNS + j;
					if (dev_board[idx] == '-') {
						Configuration c = Configuration(dev_board, lastMove(i, j, player, 0), getNMoves(), NumberStartMoves(), numberOfConfigurations);
						configurations[counter] = c;
						counter++;
						break;
					}
				}
			}
			return configurations;
		}

		return nullptr;
	}

	__device__ __host__ friend ostream& operator<<(ostream& os, const Configuration& confg) {

		for (int i = 0; i < Configuration::ROWS*Configuration::COLUMNS; i++) {
			os << confg.board[i];
			if (i % 7 == 0 && i != 0)
				os << endl;
		}
		os << endl;
		return os;
	}
};

__global__ void PrintBoard(Configuration *c)
{
	c->Configuration::PrintBoard();
}

__global__ void MiniMax(Configuration* configurations, int depth, int alpha, int beta) {

	int idx = threadIdx.x;
	int numChildren;
	if (configurations != nullptr)
		numChildren = configurations[0].numberOfConfigurations;
	else
		numChildren = 0;

	if (idx < numChildren) {

		bool isWinningMove = configurations[idx].isWinningMove();
		if (isWinningMove && configurations[idx].mLastmove.player == '0') {
			int losingScore = -(configurations[idx].getNMoves() - configurations[idx].NumberStartMoves());
			if (losingScore < beta || gBeta == 100)
				//__SM_32_ATOMIC_FUNCTIONS_H__::min(gBeta,losingScore);
				atomicMin(&gBeta, losingScore);
		}
		if (isWinningMove && configurations[idx].mLastmove.player == 'X')
		{
			int winningScore = (configurations[idx].getNMoves() - configurations[idx].NumberStartMoves());
			if (winningScore > alpha || gAlpha == -100)
				//__SM_32_ATOMIC_FUNCTIONS_H__::max(gAlpha,winningScore);
				atomicMax(&gAlpha, winningScore);
		}

		if (configurations[idx].getNMoves() > Configuration::ROWS*Configuration::COLUMNS - 1)
		{
			int drawScore = 0;
			if (drawScore > alpha || gAlpha == -100)
				//__SM_32_ATOMIC_FUNCTIONS_H__::max(gAlpha,drawScore);
				atomicMax(&gAlpha, drawScore);
		}

		//int min = -(Configuration::ROWS*Configuration::COLUMNS - 2 - configuration.getNMoves()) / 2;
		if (alpha <= gAlpha)
			alpha = gAlpha;
		else
			//__SM_32_ATOMIC_FUNCTIONS_H__::max(gAlpha,alpha);
			atomicMax(&gAlpha, alpha);

		//int max = (Configuration::ROWS*Configuration::COLUMNS - 1 - configuration.getNMoves()) / 2;
		if (beta > gBeta)
			beta = gBeta;
		else
			//__SM_32_ATOMIC_FUNCTIONS_H__::min(gBeta, beta);
			atomicMin(&gBeta, beta);

		if (gAlpha >= gBeta)
			return;

		char nextPlayer = configurations[idx].mLastmove.player == 'X' ? '0' : 'X';
		Configuration* moves = configurations[idx].GenerateNextMoves(nextPlayer);

		if (depth > 0) {
			MiniMax << <1, 1 >> >(configurations, depth - 1, alpha, beta);
		}
	}
}

int main()
{
	string line;
	double duration;
	ifstream testFile("configurations.txt");
	ofstream writeInFile;
	writeInFile.open("benchmarker.txt");

	Configuration *dev_c;
	cudaError_t cudaStatus;

	

	

	if (testFile.is_open()) {
		int i = 0;
		while (getline(testFile, line)) {

			Configuration *c = new Configuration(line);
			cudaMalloc(&dev_c, sizeof(Configuration));
			cudaMemcpy(dev_c, c, sizeof(Configuration), cudaMemcpyHostToDevice);
			Configuration *tmp;
			cudaMalloc(&tmp, sizeof(Configuration));
			cudaMemcpy(tmp, dev_c, sizeof(Configuration), cudaMemcpyDeviceToHost);


			//tmp->PrintBoard();
			/*cudaMalloc((void **)&dev_c->dev_board, 6 * sizeof(char));
			cudaMemcpy(dev_c->dev_board, c->board, 6 * sizeof(char), cudaMemcpyHostToDevice);

			for (int i = 0; i < 6; i++) {
				cudaMalloc((void **)&dev_c->dev_board[i], 7 * sizeof(char));
				cudaMemcpy(dev_c->dev_board[i], c->board[i], 7 * sizeof(char), cudaMemcpyHostToDevice);
			}*/
			
			PrintBoard<< <1, 1>> >(dev_c);

			/*writeInFile << c;
			int solution = solver.MinMax(c, 10, numeric_limits<int>::min(), numeric_limits<int>::max());

			writeInFile << "Configuration Number: " << i << endl;
			writeInFile << "Duration: " << duration << endl;
			writeInFile << "Number Of Turn Until Some Win: " << solution << endl;
			writeInFile << "Number Of Nodes Calculated: " << solver.getNodeCount() << endl;
			writeInFile << "________________________________" << endl;
			solver.ResetNodeCount();*/
			i++;
			if (i >0)
				break;
			c->deleteBoard();
		}
		testFile.close();
		writeInFile.close();
		cudaDeviceReset();
	}


	system("pause");
    return 0;
}

// Helper function for using CUDA to add vectors in parallel.
/*cudaError_t addWithCuda(int *c, const int *a, const int *b, unsigned int size)
{
    

    // Choose which GPU to run on, change this on a multi-GPU system.
    cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
        goto Error;
    }

    // Allocate GPU buffers for three vectors (two input, one output)    .
    cudaStatus = cudaMalloc((void**)&dev_c, size * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_a, size * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_b, size * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    // Copy input vectors from host memory to GPU buffers.
    cudaStatus = cudaMemcpy(dev_a, a, size * sizeof(int), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    cudaStatus = cudaMemcpy(dev_b, b, size * sizeof(int), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    // Launch a kernel on the GPU with one thread for each element.
    addKernel<<<1, size>>>(dev_c, dev_a, dev_b);

    // Check for any errors launching the kernel
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
        goto Error;
    }
    
    // cudaDeviceSynchronize waits for the kernel to finish, and returns
    // any errors encountered during the launch.
    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
        goto Error;
    }

    // Copy output vector from GPU buffer to host memory.
    cudaStatus = cudaMemcpy(c, dev_c, size * sizeof(int), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

Error:
    cudaFree(dev_c);
    cudaFree(dev_a);
    cudaFree(dev_b);
    
    return cudaStatus;
}*/
