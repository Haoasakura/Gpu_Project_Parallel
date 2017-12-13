#pragma once

#include <stdio.h>
#include <iostream>
#include <fstream>
#include <string>
#include <limits>
#include <vector>
#include <algorithm>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "sm_32_atomic_functions.h"

using namespace std;

__device__ __managed__ int gAlpha=-100;
__device__ __managed__ int gBeta=100;
//__device__ __managed__ int gScore=0;
__device__ __managed__ unsigned int nodeCount = 0;
__device__ __managed__ __int8 NUMBEROFCHILDREN = 7;

struct lastMove {
	__int8 row; __int8 column; char player;
	__device__ __host__ lastMove() {}
	__device__ __host__ lastMove(__int8 _row, __int8 _column, char _player) {
		row = _row;
		column = _column;
		player = _player;
	}
};

class Configuration {
public:
	static const __int8 ROWS = 6;  // width of the board
	static const __int8 COLUMNS = 7; // height of the board
	static const __int8 BOARD_SIZE = ROWS*COLUMNS;
	static const __int8 MIN_SCORE = -(ROWS*COLUMNS) / 2 + 3;
	static const __int8 MAX_SCORE = (ROWS*COLUMNS + 1) / 2 - 3;
	
	__int8 numberOfConfigurations=0;
	char* board;
	char* dev_board;
	lastMove mLastmove = lastMove(-1, -1, 'n');

private:
	__int8 NumberOfMoves=0;
	__int8 NumberOfStartMoves=0;

	

public:
	__device__ __host__ Configuration() {}
	__host__ Configuration(string boardConfiguration) {

		board = (char*)malloc(sizeof(char)*Configuration::BOARD_SIZE);

		for (__int8 i = 0; i <boardConfiguration.length(); i++) {
			board[i] = boardConfiguration[i];
		}

		cudaMalloc(&dev_board, sizeof(char)*Configuration::BOARD_SIZE);
		cudaMemcpy(dev_board, board, sizeof(char)*Configuration::BOARD_SIZE, cudaMemcpyHostToDevice);
		delete[] board;
		mLastmove = lastMove(-1, -1, '0');
		for each (char c in boardConfiguration)
		{
			if (c == 'X' || c == '0')
				NumberOfMoves++;
		}
		NumberOfStartMoves = NumberOfMoves;
	}

	__device__  Configuration(char* _dev_board, lastMove _move, __int8 numMoves, __int8 startMoves, __int8 numConfig) {
		mLastmove.row = _move.row;
		mLastmove.column = _move.column;
		mLastmove.player = _move.player;
		NumberOfStartMoves = startMoves;
		NumberOfMoves = numMoves + 1;
		numberOfConfigurations = numConfig;

		dev_board = (char*)malloc(sizeof(char)*Configuration::BOARD_SIZE);
		memcpy(dev_board,_dev_board,sizeof(char)*Configuration::BOARD_SIZE);
		dev_board[_move.row * COLUMNS + _move.column] = _move.player;
	}

	__device__ bool isFull() {

		__int8 idx = 0;
		__int8 counter = 0;
		for (__int8 j = 0; j < COLUMNS; j++) {
			for (__int8 i = ROWS - 1; i >= 0; i--) {
				idx = i*COLUMNS + j;
				if (dev_board[idx] == '-') {
					counter++;
					break;
				}
			}
		}

		numberOfConfigurations = counter;
		if (numberOfConfigurations > 0)
			return false;
		else
			return true;
	}

	__device__ bool isWinningMove() {

		if (mLastmove.row == -1)
			return false;
		__int8 counter = 0;
		//check the column
		for (__int8 j = 0; j < COLUMNS; j++) {
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
		for (__int8 i = 0; i < ROWS; i++) {
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
		for (__int8 k = 0; (k + mLastmove.row < ROWS && k + mLastmove.column < COLUMNS); k++) {
			if (dev_board[(mLastmove.row + k)*COLUMNS + (mLastmove.column + k)] == mLastmove.player) {
				counter++;
				if (counter >= 4)
					return true;
			}
			else
				counter = 0;
		}
		counter = 0;
		for (__int8 k = 0; (mLastmove.row - k >= 0 && mLastmove.column - k >= 0); k++) {
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
		for (__int8 k = 0; (k + mLastmove.row < ROWS && mLastmove.column - k >= 0); k++) {
			if (dev_board[(mLastmove.row + k)*COLUMNS + (mLastmove.column - k)] == mLastmove.player) {
				counter++;

				if (counter >= 4)
					return true;
			}
			else
				counter = 0;
		}
		counter = 0;
		for (__int8 k = 0; (mLastmove.row - k >= 0 && k + mLastmove.column < COLUMNS); k++) {
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
		for (__int8 i = 0; i < Configuration::ROWS; i++) {
			for (__int8 j = 0; j < Configuration::COLUMNS; j++) {
				__int8 idx = i*Configuration::COLUMNS + j;
				printf("%c", board[idx]);
			}
			printf("\n");
		}
	}

	__device__ __int8 getNMoves() {
		return NumberOfMoves;
	}

	__device__ void setNMoves(__int8 moves) {
		NumberOfMoves = moves;
	}

	__device__ __int8 NumberStartMoves()
	{
		return NumberOfStartMoves;
	}

	__device__ __host__ ~Configuration() {
	}
};
__device__ void BoardPrint(Configuration *c) {
	for (__int8 i = 0; i < Configuration::ROWS; i++) {
		for (__int8 j = 0; j < Configuration::COLUMNS; j++) {
			__int8 idx = i*Configuration::COLUMNS + j;
			printf("%c", c->dev_board[idx]);
		}
		printf("\n");
	}
}

__global__ void MiniMax(Configuration* configuration, int depth) {
	int thread = threadIdx.x;
	atomicAdd(&nodeCount,1);
	bool freeSpace = false;
	Configuration* c;
	c = (Configuration*)malloc(sizeof(Configuration));
	char nextPlayer = configuration->mLastmove.player == 'X' ? '0' : 'X';
	for (int i = Configuration::ROWS - 1; i >= 0; i--) {
		int idx = i*Configuration::COLUMNS + thread;
		if (configuration->dev_board[idx] == '-') {
			c = new Configuration(configuration->dev_board, lastMove(i, thread, nextPlayer), configuration->getNMoves(), configuration->NumberStartMoves(), configuration->numberOfConfigurations);
			freeSpace = true;
			break;
		}
	}

	if (freeSpace && thread < 7) {

		bool isWinningMove = c->isWinningMove();
		if (isWinningMove && c->mLastmove.player == '0') {
			int losingScore = -(c->getNMoves() - c->NumberStartMoves());
			if (losingScore < gBeta) {
				atomicMin(&gBeta, losingScore);
				return;
			}
		}
		if (isWinningMove && c->mLastmove.player == 'X')
		{
			int winningScore = (c->getNMoves() - c->NumberStartMoves());
			if (winningScore > gAlpha) {
				atomicMax(&gAlpha, winningScore);
				return;
			}
		}

		if (c->getNMoves() > Configuration::ROWS*Configuration::COLUMNS - 1)
		{
			int drawScore = 0;
			if (drawScore > gAlpha) {
				atomicMax(&gAlpha, drawScore);
				return;
			}
		}

		int max = (c->getNMoves() - c->NumberStartMoves());
		if (gAlpha <= max && gAlpha > 0)
			return;

		//if (gAlpha >= gBeta)
		//	return;	

		if (depth > 0 && !configuration->isFull()) {
			MiniMax << <1, 7 >> > (c, depth - 1);
			//cudaDeviceSynchronize();
		}

		/*if (nextPlayer == 'X') {
			if (depth > 0 && !configuration->isFull()) {
				MiniMax << <1, 7 >> > (c, depth - 1);
				/*if (score < alpha) {
					score = alpha;
				}*/

				/*if (score < gAlpha)
					atomicMin(&gAlpha, score);
				if (gBeta <= gAlpha) {
					gScore = score;

				}

				printf("%d  ", gAlpha);
			}
		}

		if (nextPlayer == '0') {
			if (depth > 0 && !configuration->isFull()) {
				MiniMax << <1, 7 >> > (c, depth - 1);
				//cudaDeviceSynchronize();
				/*if (score > beta) {
					score = beta;
				}*/

				/*if (score >= gBeta)
					atomicMax(&gBeta, score);
				if (gBeta <= gAlpha) {
					gScore = score;
				}

				printf("%d  ", gBeta);
			}
		}*/

	}
}

__global__ void MinMax(Configuration* configuration, __int8 depth,unsigned long numberOfNodes) {
	unsigned long idx= blockDim.x * blockIdx.x + threadIdx.x;

	if (idx < numberOfNodes) {
		//printf("%lu ", idx);

		unsigned long currentNode = idx;
		__int8* moves;
		moves = (__int8*)malloc(sizeof(__int8)*depth);
		__int8 i = 0;
		while (currentNode>7) {
			unsigned long parentNode = (int)currentNode / 7;	
			__int8 move = currentNode % 7;
			moves[i++] = move;
			i++;
			currentNode = parentNode;
		}
		moves[i] = (__int8)currentNode;

		for (__int8 m = depth - 1; m >= 0; m--) {
			for (__int8 i = Configuration::ROWS - 1; i >= 0; i--) {
				__int8 ix = i*Configuration::COLUMNS + moves[m];
				if (configuration->dev_board[ix] == '-') {
					break;
				}
				if (i == 0) {
					return;
				}
			}
		}
		Configuration* c = (Configuration*)malloc(sizeof(Configuration));
		c = new Configuration(configuration->dev_board, configuration->mLastmove, configuration->getNMoves(), configuration->NumberStartMoves(), configuration->numberOfConfigurations);

		for (__int8 m = depth-1; m>=0; m--) {
			for (__int8 i = Configuration::ROWS - 1; i >= 0; i--) {
				__int8 ix = i*Configuration::COLUMNS + moves[m];
				if (c->dev_board[ix] == '-') {
					c->dev_board[ix] = m % 2 == 0 ? 'X' : '0';
					if (m == 0)
						c->mLastmove = lastMove(i, moves[m], m % 2 == 0 ? 'X' : '0');
					break;
				}
			}
		}
		if(idx==numberOfNodes-1)
			BoardPrint(c);

		delete c;
		delete[] moves;
		

	}
}
__int8 blockDimension(unsigned long numberbOfNodes) {
	unsigned long attempt=1024;
	__int8 y = 1;
	while (attempt<numberbOfNodes)
	{
		attempt += 1024;
		y++;
	}
	return y;
}
__int8 GenerateResult(Configuration* configuration, int depth) {
	__int8 i = 0;
	gAlpha = -depth;
	gBeta = depth;
	for (; i < depth; i++)
	{
		__int8 tAlpha = gAlpha;
		unsigned long numberbOfNodes= std::pow(NUMBEROFCHILDREN, i);
		if (i < 4) {
			MinMax << <1, numberbOfNodes >> > (configuration, i,numberbOfNodes);
		}
		else {
			
			unsigned long nBlocks = blockDimension(numberbOfNodes);
			//printf("%lu ",nBlocks);
			MinMax << <nBlocks, 1024 >> > (configuration, i,numberbOfNodes);
		}
		printf("\n \n");
		cudaDeviceSynchronize();
		if (gAlpha > tAlpha) {
			return tAlpha;
		}
	}

	return gBeta;
}

int main() {
	string line;
	//double duration;
	ifstream testFile("configurations.txt");
	ofstream writeInFile;
	writeInFile.open("benchmarker.txt");
	/*size_t* s;
	s = (size_t*)malloc(sizeof(size_t));
	cudaDeviceSetLimit(cudaLimitPrintfFifoSize,16384);
	cudaDeviceGetLimit(s,cudaLimitStackSize);*/
	//GenerateResult(nullptr, 2);
	if (testFile.is_open()) {
		

		__int8 i = 0;
		while (getline(testFile, line)) {
			Configuration *dev_c, *c;
			c = (Configuration*)malloc(sizeof(Configuration));
			c = new Configuration(line);
			/*c->board = (char*)malloc(sizeof(char)*Configuration::BOARD_SIZE);

			for (int i = 0; i <line.length(); i++) {
				c->board[i] = line[i];
			}

			cudaMalloc(&c->dev_board, sizeof(char)*Configuration::BOARD_SIZE);
			cudaMemcpy(c->dev_board, c->board, sizeof(char)*Configuration::BOARD_SIZE, cudaMemcpyHostToDevice);*/
			cudaMalloc(&dev_c, sizeof(Configuration));
			cudaMemcpy(dev_c, c, sizeof(Configuration), cudaMemcpyHostToDevice);
			GenerateResult(dev_c, 7);
			//BoardPrint << <1, 1 >> > (dev_c);
			//MiniMax << <1, 7 >> > (dev_c,3);
			cudaDeviceSynchronize();

			//cudaMemcpy(&alpha,&gAlpha,sizeof(int),cudaMemcpyDeviceToHost);
			//cudaMemcpy(&beta, &gBeta, sizeof(int), cudaMemcpyDeviceToHost);
			//cudaMemcpy(&score, &gScore, sizeof(int), cudaMemcpyDeviceToHost);
			printf("Configuration N  %d \n", i);
			printf("galpha %d \n", gAlpha);
			printf("gbeta %d \n", gBeta);
			//printf("gscore %d \n", gScore);
			printf("nodes %u \n", nodeCount);
			printf("-----------------------------------\n");
			
			free(c);
			cudaFree(dev_c);
			gAlpha = -100;
			gBeta = 100;
			//gScore = 0;
			nodeCount = 0;
			cudaDeviceReset();
			i++;
			if (i >0)
				break;
		}
	}

	
	system("pause");
    return 0;
}

