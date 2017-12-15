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

__device__ __managed__ int gAlpha;
__device__ __managed__ int gBeta;
__device__ __managed__ unsigned int nodeCount = 0;
__device__ __managed__ __int8 NUMBEROFCHILDREN = 7;

struct lastMove {
	__int8 row; __int8 column; char player; __int8 value;
	__device__ __host__ lastMove() {}
	__device__ __host__ lastMove(__int8 _row, __int8 _column, char _player,__int8 _value) {
		row = _row;
		column = _column;
		player = _player;
		value = _value;
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
	lastMove mLastmove = lastMove(-1, -1, 'n',0);

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
		PrintBoard();
		printf("\n");
		delete[] board;
		mLastmove = lastMove(-1, -1, '0',0);
		for each (char c in boardConfiguration)
		{
			if (c == 'X' || c == '0')
				NumberOfMoves++;
		}
		NumberOfStartMoves = NumberOfMoves;
	}
	__host__ Configuration::Configuration(char* _board, lastMove _move, int numMoves, int startMoves) {
		_board[_move.row*COLUMNS + _move.column] = _move.player;
		mLastmove.row = _move.row;
		mLastmove.column = _move.column;
		mLastmove.player = _move.player;
		NumberOfStartMoves = startMoves;
		NumberOfMoves = numMoves + 1;
		board = new char[ROWS*COLUMNS];
		for (int i = 0; i < ROWS*COLUMNS; i++) {
			board[i] = _board[i];
		}

		cudaMalloc(&dev_board, sizeof(char)*Configuration::BOARD_SIZE);
		cudaMemcpy(dev_board, board, sizeof(char)*Configuration::BOARD_SIZE, cudaMemcpyHostToDevice);
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
		int counter = 0;

		//check the column
		for (int j = 0; j < COLUMNS; j++) {
			if (board[mLastmove.row*COLUMNS + j] == mLastmove.player) {
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
			if (board[i*COLUMNS + mLastmove.column] == mLastmove.player) {
				counter++;
				if (counter >= 4)
					return true;
			}
			else
				counter = 0;
		}
		counter = 0;
		//check right diagonal
		int ki = mLastmove.row;
		int ky = mLastmove.column;
		for (; (ki > 0 && ky > 0); ki--, ky--) {
		}
		if (!(ki >= 3 || ky >= 3)) {
			for (int k = 0; (ki + k < ROWS && ky + k < COLUMNS); k++) {
				if (board[(ki + k)*COLUMNS + (ky + k)] == mLastmove.player) {
					counter++;
					if (counter >= 4) {
						return true;
					}
				}
				else {
					counter = 0;
				}
			}
		}

		//check left diagonal
		counter = 0;
		ki = mLastmove.row;
		ky = mLastmove.column;
		for (; (ki < ROWS - 1 && ky > 0); ki++, ky--) {
		}
		if (!(ki < 3 || ky > 3)) {
			for (int k = 0; (ki - k >= 0 && ky + k < COLUMNS); k++) {
				if (board[(ki - k)*COLUMNS + (ky + k)] == mLastmove.player) {
					counter++;
					if (counter >= 4) {
						return true;
					}
				}
				else {
					counter = 0;
				}
			}
		}
		return false;
	}
	
	char* Configuration::getBoard() {
		char * _board = new char[ROWS*COLUMNS];

		for (int i = 0; i < ROWS*COLUMNS; i++) {
			_board[i] = board[i];
		}
		return _board;
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


	int Configuration::ValutateMove(lastMove mLastmove, int pawnInARow) {
		int value = 0;
		int _value = 0;
		bool yourMove = false;
		if (mLastmove.row == -1)
			return value;

		int counter = 0;
		//check the rows
		for (int j = 0; j < COLUMNS; j++) {
			if (board[mLastmove.row*COLUMNS + j] == mLastmove.player) {
				counter++;
				if (j == mLastmove.column) {
					yourMove = true;
				}
				if (counter >= pawnInARow && yourMove) {
					if (counter == 4) {
						_value = counter + 5;
						break;
					}
					_value = counter;
				}
			}
			else {
				counter = 0;
				yourMove = false;
			}
		}
		value += _value;
		_value = 0;
		counter = 0;
		//check the columns
		for (int i = mLastmove.row; i < ROWS; i++) {
			if (board[i*COLUMNS + mLastmove.column] == mLastmove.player) {
				counter++;

				if (counter >= pawnInARow) {
					if (counter == 4) {
						_value = counter + 5;
						break;
					}
					_value = counter;
				}
			}
			else
				break;
		}

		value += _value;
		_value = 0;
		//value += (mLastmove.row<4)?mLastmove.row:3;
		counter = 0;
		yourMove = false;
		//check right diagonal
		int ki = mLastmove.row;
		int ky = mLastmove.column;
		for (; (ki > 0 && ky > 0); ki--, ky--) {
		}
		if (!(ki >= 3 || ky >= 3)) {
			for (int k = 0; (ki + k < ROWS && ky + k < COLUMNS); k++) {
				if (board[(ki + k)*COLUMNS + (ky + k)] == mLastmove.player) {
					counter++;
					if ((ki + k) == mLastmove.row && (ky + k) == mLastmove.column) {
						yourMove = true;
					}
					if (counter >= pawnInARow && yourMove) {
						if (counter == 4) {
							_value = counter + 5;
							break;
						}
						_value = counter;
					}
				}
				else {
					counter = 0;
					yourMove = false;
				}
			}
		}
		value += _value;
		_value = 0;
		counter = 0;
		yourMove = false;
		//check left diagonal
		ki = mLastmove.row;
		ky = mLastmove.column;
		for (; (ki < ROWS - 1 && ky > 0); ki++, ky--) {
		}
		if (!(ki < 3 || ky > 3)) {
			for (int k = 0; (ki - k >= 0 && ky + k < COLUMNS); k++) {
				if (board[(ki - k)*COLUMNS + (ky + k)] == mLastmove.player) {
					counter++;
					if ((ki - k) == mLastmove.row && (ky + k) == mLastmove.column) {
						yourMove = true;
					}
					if (counter >= pawnInARow && yourMove) {
						if (counter == 4) {
							_value = counter + 5;
							break;
						}
						_value = counter;
					}
				}
				else {
					counter = 0;
					yourMove = false;
				}
			}
		}
		value += _value;

		return value;
	}

	int Configuration::ValutateEnemyPositions(lastMove mLastmove, int pawnInARow) {
		int value = 0;
		if (mLastmove.row == -1)
			return value;

		int counter = 0;
		bool blocked = false;
		//check the rows
		for (int j = 0; j < COLUMNS; j++) {
			if ((board[mLastmove.row*COLUMNS + j] != mLastmove.player && board[mLastmove.row*COLUMNS + j] != '-') || j == mLastmove.column) {
				counter++;
				if (j == mLastmove.column)
					blocked = true;
				if (counter >= pawnInARow) {
					if (blocked) {
						value += pawnInARow + 3;
						break;
					}
				}
			}
			else
				counter = 0;
		}

		//check the Columns
		counter = 0;
		blocked = false;
		for (int i = 0; i < ROWS; i++) {
			if ((board[i*COLUMNS + mLastmove.column] != mLastmove.player && board[i*COLUMNS + mLastmove.column] != '-') || i == mLastmove.row) {
				counter++;
				if (i == mLastmove.row)
					blocked = true;
				if (counter >= pawnInARow) {
					if (blocked) {
						value += pawnInARow + 3;
						break;
					}
				}
			}
			else
				counter = 0;
		}

		//check right diagonal
		int ki = mLastmove.row;
		int ky = mLastmove.column;
		counter = 0;
		blocked = false;
		for (; (ki > 0 && ky > 0); ki--, ky--) {
		}
		if (!(ki >= 3 || ky >= 3)) {
			for (int k = 0; (ki + k < ROWS && ky + k < COLUMNS); k++) {
				if ((board[(ki + k)*COLUMNS + (ky + k)] != mLastmove.player && board[(ki + k)*COLUMNS + (ky + k)] != '-') || ((ki + k) == mLastmove.row && (ky + k) == mLastmove.column)) {
					counter++;
					if ((ki + k) == mLastmove.row && (ky + k) == mLastmove.column)
						blocked = true;
					if (counter >= pawnInARow) {
						if (blocked) {
							value += pawnInARow + 3;
							break;
						}
					}
				}
				else
					counter = 0;
			}
		}

		//check left diagonal
		counter = 0;
		blocked = false;
		ki = mLastmove.row;
		ky = mLastmove.column;
		for (; (ki < ROWS - 1 && ky > 0); ki++, ky--) {
		}
		if (!(ki < 3 || ky > 3)) {
			for (int k = 0; (ki - k >= 0 && ky + k < COLUMNS); k++) {
				if ((board[(ki - k)*COLUMNS + (ky + k)] != mLastmove.player && board[(ki - k)*COLUMNS + (ky + k)] != '-') || ((ki - k) == mLastmove.row && (ky + k) == mLastmove.column)) {
					counter++;
					if ((ki - k) == mLastmove.row && (ky + k) == mLastmove.column)
						blocked = true;
					if (counter >= pawnInARow) {
						if (blocked) {
							value += pawnInARow + 3;
							break;
						}
					}
				}
				else
					counter = 0;
			}
		}

		return value;
	}
	//genera le sette mosse successive
	vector<lastMove> Configuration::GenerateNextMoves(char player) {

		vector<lastMove> moves = vector<lastMove>();
		int idx = 0;
		for (int j = 0; j < COLUMNS; j++) {
			for (int i = ROWS - 1; i >= 0; i--) {
				idx = i*COLUMNS + j;
				if (board[idx] == '-') {
					moves.push_back(lastMove(i, j, player, 0));
					break;
				}
			}
		}
		return SortNextMoves(moves);
	}

	vector<lastMove> Configuration::SortNextMoves(vector<lastMove> moves) {
		bool bDone = false;

		for (int i = 0; i < moves.size(); i++) {
			lastMove tmp = moves[i];
			moves[i] = moves[moves.size() / 2 + (1 - 2 * (i % 2))*(i + 1) / 2];
			moves[moves.size() / 2 + (1 - 2 * (i % 2))*(i + 1) / 2] = tmp;
		}


		for (int i = 0; i < moves.size(); ++i) {
			board[moves[i].row*COLUMNS + moves[i].column] = moves[i].player;
			moves[i].value = ValutateMove(moves[i], 2) + ValutateEnemyPositions(moves[i], 4);
			board[moves[i].row*COLUMNS + moves[i].column] = '-';
		}

		while (!bDone) {
			bDone = true;
			for (int i = 0; i < moves.size() - 1; ++i) {
				if (moves[i].value < moves[i + 1].value) {
					lastMove tmp = moves[i];
					moves[i] = moves[i + 1];
					moves[i + 1] = tmp;
					bDone = false;
				}
			}
		}
		return moves;
	}

	__device__ __host__ __int8 getNMoves() {
		return NumberOfMoves;
	}

	__device__ __host__ void setNMoves(__int8 moves) {
		NumberOfMoves = moves;
	}

	__device__ __host__ __int8 NumberStartMoves()
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

int Pv_Split(Configuration* configuration, int depth,int alpha,int beta) {
	nodeCount++;
	bool isWinningMove = configuration->isWinningMove();
	if ((isWinningMove && configuration->mLastmove.player == '0') || depth == 0) {
		return -(configuration->getNMoves() - configuration->NumberStartMoves());
	}

	if ((isWinningMove && configuration->mLastmove.player == 'X') || depth == 0) {
		return (configuration->getNMoves() - configuration->NumberStartMoves());
	}

	if (configuration->getNMoves() > Configuration::ROWS*Configuration::COLUMNS - 1)
		return 0;

	char nextPlayer = configuration->mLastmove.player == 'X' ? '0' : 'X';
	vector<lastMove> moves = configuration->GenerateNextMoves(nextPlayer);
	
	int* score; int* dev_score;
	score= (int*)malloc(sizeof(int));
	cudaMalloc(&dev_score, sizeof(int));
	
		Configuration* c = (Configuration*)malloc(sizeof(Configuration));

			c = new Configuration(configuration->getBoard(), moves[0], configuration->getNMoves(), configuration->NumberStartMoves(), configuration->numberOfConfigurations);

			*score = -Pv_Split(c, depth - 1, -beta, -alpha);
			cudaMemcpy(dev_score, score, sizeof(int), cudaMemcpyHostToDevice);

			Configuration* dev_c;
			c = new Configuration(configuration->dev_board, moves[0], configuration->getNMoves(), configuration->NumberStartMoves(), configuration->numberOfConfigurations);
			cudaMalloc(&dev_c, sizeof(Configuration));
			cudaMemcpy(dev_c, c, sizeof(Configuration), cudaMemcpyHostToDevice);
			<< < >> > ();
			cudaDeviceSynchronize();
			cudaMemcpy(score, dev_score, sizeof(int), cudaMemcpyDeviceToHost);
			//da mettere nel kernel
			//score = -Pvs(dev_c, depth - 1, (-alpha - 1), -alpha);

			if (alpha < *score < beta) {
				<< < >> > ();
				cudaDeviceSynchronize();
				cudaMemcpy(score, dev_score, sizeof(int), cudaMemcpyDeviceToHost);
				//score = -Pvs(dev_c, depth - 1, -beta, -score);
			}
				
		alpha = max(alpha, *score);
		if (alpha >= beta) {
			delete c;
			return alpha;
		}
		delete c;

	moves.clear();
	moves.shrink_to_fit();

	return alpha;
}

__global__ void MinMax(Configuration* configuration, __int8 depth,unsigned long numberOfNodes) {
	unsigned long idx= blockDim.x * blockIdx.x + threadIdx.x;

	if (idx < numberOfNodes) {
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
					c->dev_board[ix] = depth % 2 == 0 ? '0' : 'X';
					if (m == 0)
						c->mLastmove = lastMove(i, moves[m], depth % 2 == 0 ? '0' : 'X');
					break;
				}
			}
		}

		bool isWinningMove = c->isWinningMove();
		if (isWinningMove && c->mLastmove.player == '0') {
			int losingScore = -depth;
			if (losingScore < gBeta) {
				//BoardPrint(c);
				atomicMin(&gBeta, losingScore);
				return;
			}
		}
		if (isWinningMove && c->mLastmove.player == 'X')
		{
			int winningScore = depth;
			if (winningScore > gAlpha) {
				atomicMax(&gAlpha, winningScore);
				//BoardPrint(c);
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

		int max = depth;
		if (gAlpha <= max && gBeta >= max) {
			return;
		}

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
		cudaDeviceSynchronize();
		if (gAlpha > tAlpha) {
			return gAlpha;
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
			int r=GenerateResult(dev_c, 7);
			
			//MiniMax << <1, 7 >> > (dev_c,3);
			cudaDeviceSynchronize();

			//cudaMemcpy(&alpha,&gAlpha,sizeof(int),cudaMemcpyDeviceToHost);
			//cudaMemcpy(&beta, &gBeta, sizeof(int), cudaMemcpyDeviceToHost);
			//cudaMemcpy(&score, &gScore, sizeof(int), cudaMemcpyDeviceToHost);
			printf("Configuration N  %d \n", i);
			printf("galpha %d \n", gAlpha);
			printf("gbeta %d \n", gBeta);
			printf("Result %d \n", r);
			printf("nodes %u \n", nodeCount);
			printf("-----------------------------------\n");
			
			free(c);
			cudaFree(dev_c);
			gAlpha = -100;
			gBeta = 100;
			nodeCount = 0;
			cudaDeviceReset();
			i++;
			if (i >1)
				break;
		}
	}

	
	system("pause");
    return 0;
}

