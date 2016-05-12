#! /usr/bin/env python


#    Solving a 9x9 Sudoku puzzle (45 numbers missing).
#
#    Copyright (C) 2013 Efstathios Chatzikyriakidis <contact@efxa.org>
#
#    This program is free software: you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation, either version 3 of the License, or
#    (at your option) any later version.
#
#    This program is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.
#
#    You should have received a copy of the GNU General Public License
#    along with this program.  If not, see <http://www.gnu.org/licenses/>.


try:
	from SudokuSolver import SudokuGA
except (ImportError) as error:
	import sys, os
	sys.exit('{0}: {1}.'.format(os.path.basename(__file__), error))


parameters = {
	'populationSize': 6000,

	'mutationProbability': 0.012,

	'tournamentSize': 8,

	'candidateGenotype': [[2, 0, 0, 8, 0, 4, 0, 7, 0],
			      [0, 8, 9, 0, 0, 1, 0, 0, 3],
			      [0, 1, 0, 3, 6, 0, 0, 2, 0],
			      [0, 6, 0, 0, 2, 0, 1, 9, 0],
			      [0, 2, 5, 0, 0, 0, 7, 0, 8],
			      [7, 0, 3, 0, 8, 0, 6, 0, 0],
			      [6, 0, 0, 5, 0, 0, 0, 1, 2],
			      [0, 9, 0, 0, 1, 0, 4, 5, 0],
			      [0, 0, 4, 0, 3, 6, 0, 0, 8]]
}

ga = SudokuGA(parameters)

solution = ga.evolution()

print solution
