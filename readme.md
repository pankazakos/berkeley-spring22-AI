# Berkeley pacman projects 0-3

## Setup
```
conda create -n AI python=3.6.10
conda activate AI
pip install pycosat
```

## Run autograder for each project
```
cd project_name
python autograder.py
```

## 1.Search
### pacman game
```
python pacman.py
```
### Question 1 (3 points): Finding a Fixed Food Dot using Depth First Search
```
python pacman.py -l tinyMaze -p SearchAgent
python pacman.py -l mediumMaze -p SearchAgent
python pacman.py -l bigMaze -z .5 -p SearchAgent
```
### Question 2 (3 points): Breadth First Search
```
python pacman.py -l mediumMaze -p SearchAgent -a fn=bfs
python pacman.py -l bigMaze -p SearchAgent -a fn=bfs -z .5
```
### Question 3 (3 points): Varying the Cost Function
```
python pacman.py -l mediumMaze -p SearchAgent -a fn=ucs
python pacman.py -l mediumDottedMaze -p StayEastSearchAgent
python pacman.py -l mediumScaryMaze -p StayWestSearchAgent
```
### Question 4 (3 points): A* search
```
python pacman.py -l bigMaze -z .5 -p SearchAgent -a fn=astar,heuristic=manhattanHeuristic
```
### Question 5 (3 points): Finding All the Corners
```
python pacman.py -l tinyCorners -p SearchAgent -a fn=bfs,prob=CornersProblem
python pacman.py -l mediumCorners -p SearchAgent -a fn=bfs,prob=CornersProblem
```
### Question 6 (3 points): Corners Problem: Heuristic
```
python pacman.py -l mediumCorners -p AStarCornersAgent -z 0.5
```
### Question 7 (4 points): Eating All The Dots
```
python pacman.py -l testSearch -p AStarFoodSearchAgent
```
### Question 8 (3 points): Suboptimal Search
```
python pacman.py -l bigSearch -p ClosestDotSearchAgent -z .5
```

## 2.Multiagent
### Question 1 (4 points): Reflex Agent
```
python pacman.py -p ReflexAgent -l testClassic
```
### Question 2 (5 points): Minimax
```
python pacman.py -p MinimaxAgent -l minimaxClassic -a depth=4
python pacman.py -p MinimaxAgent -l trappedClassic -a depth=3
```
### Question 3 (5 points): Alpha-Beta Pruning
```
python pacman.py -p AlphaBetaAgent -a depth=3 -l smallClassic
```
### Question 4 (5 points): Expectimax
```
python pacman.py -p ExpectimaxAgent -l minimaxClassic -a depth=3
```
### Question 5 (6 points): Evaluation Function
```
python autograder.py -q q5
```

## 3.Logic
```
python autograder.py
```

## Results
```
Project0: 3/3
Project1: 26/25
Project2: 25/25
Project3: 26/26
```