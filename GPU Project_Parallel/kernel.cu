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
__device__ __managed__ int NUMBEROFCHILDREN = 7;

struct lastMove {
	int row; int column; char player; int value;
	__device__ __host__ lastMove() {}
	__device__ __host__ lastMove(int _row, int _column, char _player, int _value) {
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
	static const int BOARD_SIZE = ROWS*COLUMNS;
	static const int MIN_SCORE = -(ROWS*COLUMNS) / 2 + 3;
	static const int MAX_SCORE = (ROWS*COLUMNS + 1) / 2 - 3;
	
	int numberOfConfigurations=0;
	char* board;
	char* dev_board;
	lastMove mLastmove = lastMove(-1, -1, 'n', 0);

private:
	int NumberOfMoves=0;
	int NumberOfStartMoves=0;

	

public:
	__device__ __host__ Configuration() {}
	__host__ Configuration(string boardConfiguration) {

		board = (char*)malloc(sizeof(char)*Configuration::BOARD_SIZE);

		for (int i = 0; i <boardConfiguration.length(); i++) {
			board[i] = boardConfiguration[i];
		}

		cudaMalloc(&dev_board, sizeof(char)*Configuration::BOARD_SIZE);
		cudaMemcpy(dev_board, board, sizeof(char)*Configuration::BOARD_SIZE, cudaMemcpyHostToDevice);

		mLastmove = lastMove(-1, -1, '0', 0);
		for each (char c in boardConfiguration)
		{
			if (c == 'X' || c == '0')
				NumberOfMoves++;
		}
		NumberOfStartMoves = NumberOfMoves;
	}

	__device__  Configuration(char* _dev_board, lastMove _move, int numMoves, int startMoves, int numConfig) {
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
		if (numberOfConfigurations > 0)
			return false;
		else
			return true;
	}

	__device__ bool isWinningMove() {

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
				int idx = i*Configuration::COLUMNS + j;
				printf("%c", board[idx]);
			}
			printf("\n");
		}
	}

	__device__ int getNMoves() {
		return NumberOfMoves;
	}

	__device__ void setNMoves(int moves) {
		NumberOfMoves = moves;
	}

	__device__ int NumberStartMoves()
	{
		return NumberOfStartMoves;
	}

	__device__ __host__ ~Configuration() {
	}
};
__device__ void BoardPrint(Configuration *c) {
	for (int i = 0; i < Configuration::ROWS; i++) {
		for (int j = 0; j < Configuration::COLUMNS; j++) {
			int idx = i*Configuration::COLUMNS + j;
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
			c = new Configuration(configuration->dev_board, lastMove(i, thread, nextPlayer, 0), configuration->getNMoves(), configuration->NumberStartMoves(), configuration->numberOfConfigurations);
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

__global__ void MinMax(Configuration* configuration, int depth,unsigned long numberOfNodes) {
	unsigned long idx= blockDim.x * blockIdx.x + threadIdx.x;

	if (idx < numberOfNodes) {
		//printf("%lu ", idx);
		Configuration* c = (Configuration*)malloc(sizeof(Configuration));
		c = new Configuration(configuration->dev_board, configuration->mLastmove, configuration->getNMoves(), configuration->NumberStartMoves(), configuration->numberOfConfigurations);

		unsigned long currentNode = idx;
		int* moves;
		moves = (int*)malloc(sizeof(int)*depth);
		int i = 0;
		while (currentNode>7) {
			int parentNode = (int)currentNode / 7;	
			int move = currentNode % 7;
			moves[i++] = move;
			i++;
			currentNode = parentNode;
		}
		moves[i] = currentNode;

		for (int m = depth-1; m>=0; m--) {
			for (int i = Configuration::ROWS - 1; i >= 0; i--) {
				int ix = i*Configuration::COLUMNS + moves[m];
				if (c->dev_board[ix] == '-') {
					c->dev_board[ix] = m % 2 == 0 ? 'X' : '0';
					if (m == 0)
						c->mLastmove = lastMove(i, moves[m], m % 2 == 0 ? 'X' : '0', 0);
					break;
				}
				if (i == 0) {
					return;
				}
			}
		}
		if(idx==numberOfNodes-1)
			BoardPrint(c);

		delete c;
		delete[] moves;
		

	}
}
int blockDimension(unsigned long numberbOfNodes) {
	unsigned long attempt=1024;
	int y = 1;
	while (attempt<numberbOfNodes)
	{
		attempt += 1024;
		y++;
	}
	return y;
}
int GenerateResult(Configuration* configuration, int depth) {
	int i = 0;
	gAlpha = -depth;
	gBeta = depth;
	for (; i < depth; i++)
	{
		int tAlpha = gAlpha;
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
		

		int i = 0;
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
			GenerateResult(dev_c, 3);
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

