#pragma once

#include <stdio.h>
#include <iostream>
#include <fstream>
#include <string>
#include <limits>
#include <cstdint>
#include <cassert>
#include <vector>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

using namespace std;

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

	char** board;
	char* dev_board;

private:
	int NumberOfMoves;

	int NumberOfStartMoves;

	lastMove mLastmove = lastMove(-1, -1, 'n', 0);

public:
	Configuration(string boardConfiguration) {
		SetupBoard(boardConfiguration);

		cudaMalloc((void **)&dev_board, 6 * 7 * sizeof(char*));
		cudaMemcpy(dev_board, board[0], 6 * 7 * sizeof(char*), cudaMemcpyHostToDevice);

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

	__device__ __host__ Configuration(char** _board, lastMove _move, int numMoves, int startMoves) {
		_board[_move.row][_move.column] = _move.player;
		mLastmove.row = _move.row;
		mLastmove.column = _move.column;
		mLastmove.player = _move.player;
		NumberOfStartMoves = startMoves;
		NumberOfMoves = numMoves + 1;
		board = new char*[ROWS];
		for (int i = 0; i < ROWS; i++) {
			board[i] = new char[COLUMNS];
		}

		for (int i = 0; i < ROWS; i++) {
			for (int j = 0; j < COLUMNS; j++) {
				board[i][j] = _board[i][j];
			}

		}
	}



	__device__ __host__ bool isWinningMove() {
		//cout << mLastmove.row << "   " << mLastmove.column << endl;
		if (mLastmove.row == -1)
			return false;
		int counter = 0;
		//check the column
		for (int j = 0; j < COLUMNS; j++) {
			if (board[mLastmove.row][j] == mLastmove.player) {
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
			if (board[i][mLastmove.column] == mLastmove.player) {
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
			if (board[mLastmove.row + k][mLastmove.column + k] == mLastmove.player) {
				counter++;
				if (counter >= 4)
					return true;
			}
			else
				counter = 0;
		}
		counter = 0;
		for (int k = 0; (mLastmove.row - k >= 0 && mLastmove.column - k >= 0); k++) {
			if (board[mLastmove.row - k][mLastmove.column - k] == mLastmove.player) {
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
			if (board[mLastmove.row + k][mLastmove.column - k] == mLastmove.player) {
				counter++;

				if (counter >= 4)
					return true;
			}
			else
				counter = 0;
		}
		counter = 0;
		for (int k = 0; (mLastmove.row - k >= 0 && k + mLastmove.column < COLUMNS); k++) {
			if (board[mLastmove.row - k][mLastmove.column + k] == mLastmove.player) {
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
				printf("%c",board[i][j]);
			}
			printf("\n");
		}
	}

	__device__ __host__ char** getBoard() {
		char ** _board = new char*[ROWS];
		for (int i = 0; i < ROWS; i++)
		{
			_board[i] = new char[COLUMNS];
		}

		for (int i = 0; i < ROWS; i++) {
			for (int j = 0; j < COLUMNS; j++) {
				_board[i][j] = board[i][j];
			}
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
		/*for (int i = 0; i < ROWS; i++) {
			delete[] board[i];
		}*/
		delete[] board;
	}

	__device__ __host__ ~Configuration() {
	}

private:

	__device__ __host__ void SetupBoard(string boardConfiguration) {
		board = new char*[ROWS];
		board[0] = new char[ROWS*COLUMNS];
		for (int i = 1; i < ROWS; ++i) {
			board[i] = board[i - 1] + COLUMNS;
		}
		for (int i = 0; i < (boardConfiguration.length() / 7); i++) {
			int muliply = i * 7;
			for (int j = 0; j < 7; j++) {
				board[i % 6][j] = boardConfiguration[muliply + j];
			}
		}
	}

	__device__ __host__ int ValutateMove(lastMove mLastmove, int pawnInARow) {
		int value = 0;
		if (mLastmove.row == -1)
			return value;

		int counter = 0;
		//check the column
		for (int j = (mLastmove.column - pawnInARow > 0 ? mLastmove.column - pawnInARow : 0); j < (mLastmove.column + pawnInARow < COLUMNS ? mLastmove.column + pawnInARow : COLUMNS); j++) {
			if (board[mLastmove.row][j] == mLastmove.player) {
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
			if (board[i][mLastmove.column] == mLastmove.player) {
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
			if (board[mLastmove.row + k][mLastmove.column + k] == mLastmove.player) {
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
			if (board[mLastmove.row - k][mLastmove.column - k] == mLastmove.player) {
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
			if (board[mLastmove.row + k][mLastmove.column - k] == mLastmove.player) {
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
			if (board[mLastmove.row - k][mLastmove.column + k] == mLastmove.player) {
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
				if (board[i][j] != mLastmove.player &&board[i][j] != '-') {
					counter++;
					if (counter >= pawnInARow) {
						if (j - counter >= 1) {
							if (board[i][j - counter] == '-')
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
				if (board[i][j] != mLastmove.player &&board[i][j] != '-') {
					counter++;
					if (counter >= pawnInARow) {
						if (i - counter >= 1) {
							if (board[i - counter][j] == '-')
								value -= pawnInARow;
						}
						if (i < ROWS - 1) {
							if (board[i + 1][j] == '-')
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
	__device__ __host__ vector<lastMove> GenerateNextMoves(char player) {

		vector<lastMove> moves = vector<lastMove>();
		for (int j = 0; j < COLUMNS; j++) {
			for (int i = ROWS - 1; i >= 0; i--) {
				if (board[i][j] == '-') {
					moves.push_back(lastMove(i, j, player, 0));
					break;
				}
			}
		}
		return SortNextMoves(moves);

	}

	__device__ __host__ vector<lastMove> SortNextMoves(vector<lastMove> moves) {
		bool bDone = false;
		for (int i = 0; i < moves.size(); ++i) {
			board[moves[i].row][moves[i].column] = moves[i].player;
			moves[i].value = ValutateMove(moves[i], 2) + ValutateMove(moves[i], 3);
			board[moves[i].row][moves[i].column] = '-';
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

	__device__ __host__ friend ostream& operator<<(ostream& os, const Configuration& confg) {

		for (int i = 0; i < Configuration::ROWS; i++) {
			for (int j = 0; j < Configuration::COLUMNS; j++) {
				os << confg.board[i][j];
			}
			os << endl;
		}

		return os;
	}


};

__global__ void PrintBoard(Configuration *c)
{

	//c->Configuration::PrintBoard();
	//printf("%d", c->NumberStartMoves());
	for (int i = 0; i < Configuration::ROWS; i++) {
		for (int j = 0; j < Configuration::COLUMNS; j++) {
			int idx = i*Configuration::COLUMNS + j;
			printf("%c", c->dev_board[idx]);
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
