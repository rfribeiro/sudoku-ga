#! /usr/bin/env python


#    Solving a 4x4 Sudoku puzzle (11 numbers missing).
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
	'populationSize': 150,

	'mutationProbability': 0.012,

	'tournamentSize': 2,

	'candidateGenotype': [[0, 0, 3, 0],
			      [0, 4, 0, 0],
			      [0, 0, 0, 2],
			      [4, 0, 1, 0]]
}

ga = SudokuGA(parameters)

solution = ga.evolution()

print solution
