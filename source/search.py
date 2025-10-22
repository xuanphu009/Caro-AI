import math
import sys
import source.utils as utils
import torch
from source.model import load_model_into_cache, evaluate_model

sys.setrecursionlimit(1500)

N = 15 # board size 15x15

class GomokuAI():
    def __init__(self, depth=4):
        self.depth = depth # default depth set to 4
        self.boardMap = [[0 for j in range(N)] for i in range(N)]
        self.currentI = -1
        self.currentJ = -1
        self.nextBound = {} # to store possible moves to be checked (i,j)
        self.boardValue = 0 

        self.turn = 0 
        self.lastPlayed = 0
        self.emptyCells = N * N
        self.patternDict = utils.create_pattern_dict() # dictionary containing all patterns with corresponding score
        
        self.zobristTable = utils.init_zobrist()
        self.rollingHash = 0
        self.TTable = {}
        
        # Killer move heuristic - track moves that caused cutoffs
        self.killerMoves = [[] for _ in range(depth + 5)]
        
        # Cache for pattern evaluations to avoid redundant calculations
        self.evalCache = {}

    # Draw board in string format
    def drawBoard(self):
        '''
        States:
        0 = empty (.)
        1 = AI (x)
        -1 = human (o)
        '''
        for i in range(N):
            for j in range(N):
                if self.boardMap[i][j] == 1:
                    state = 'x'
                if self.boardMap[i][j] == -1:
                    state = 'o'
                if self.boardMap[i][j] == 0:
                    state = '.'
                print('{}|'.format(state), end=" ")
            print()
        print() 
    
    # Check whether a move is inside the board and whether it is empty
    def isValid(self, i, j, state=True):
        '''
        if state=True, check also whether the position is empty
        if state=False, only check whether the move is inside the board
        '''
        if i<0 or i>=N or j<0 or j>=N:
            return False
        if state:
            if self.boardMap[i][j] != 0:
                return False
            else:
                return True
        else:
            return True

    # Given a position, change the state and "play" the move
    def setState(self, i, j, state):
        '''
        States:
        0 = empty (.)
        1 = AI (x)
        -1 = human (o)
        '''
        assert state in (-1,0,1), 'The state inserted is not -1, 0 or 1'
        self.boardMap[i][j] = state
        self.lastPlayed = state


    def countDirection(self, i, j, xdir, ydir, state):
        count = 0
        # look for 4 more steps on a certain direction
        for step in range(1, 5): 
            if xdir != 0 and (j + xdir*step < 0 or j + xdir*step >= N): # ensure move inside the board
                break
            if ydir != 0 and (i + ydir*step < 0 or i + ydir*step >= N):
                break
            if self.boardMap[i + ydir*step][j + xdir*step] == state:
                count += 1
            else:
                break
        return count

    # Check whether there are 5 pieces connected (in all 4 directions)
    def isFive(self, i, j, state):
        # 4 directions: horizontal, vertical, 2 diagonals
        directions = [[(-1, 0), (1, 0)], \
                      [(0, -1), (0, 1)], \
                      [(-1, 1), (1, -1)], \
                      [(-1, -1), (1, 1)]]
        for axis in directions:
            axis_count = 1
            for (xdir, ydir) in axis:
                axis_count += self.countDirection(i, j, xdir, ydir, state)
                if axis_count >= 5:
                    return True
        return False

    # Return all possible child moves (i,j) in a board status given the bound
    # Sorted in descending order based on their value (better moves first)
    def childNodes(self, bound, depth):
        """
        Improved move ordering:
        1. Killer moves (moves that caused cutoffs at this depth)
        2. Moves sorted by their heuristic value
        """
        moves = []
        killer_set = set(self.killerMoves[depth]) if depth < len(self.killerMoves) else set()
        
        # Add killer moves first
        for move in killer_set:
            if move in bound:
                moves.append((move, bound[move] + 50000))  # Bonus for killer moves
        
        # Add remaining moves sorted by value
        for pos, val in bound.items():
            if pos not in killer_set:
                moves.append((pos, val))
        
        # Sort by value (descending) - best moves first for better pruning
        moves.sort(key=lambda x: x[1], reverse=True)
        
        # Yield only the positions
        for move, _ in moves:
            yield move

    # Update boundary for new possible moves given the recently played move
    def updateBound(self, new_i, new_j, bound):
        # get rid of the played position
        played = (new_i, new_j)
        if played in bound:
            bound.pop(played)
        # check in all 8 directions - but only add neighbors closer to stones
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, 1), (1, -1), (-1, -1), (1, 1)]
        for dir in directions:
            # Check 1-2 steps away for better coverage
            for step in [1, 2]:
                new_col = new_j + dir[0] * step
                new_row = new_i + dir[1] * step
                if self.isValid(new_row, new_col) and (new_row, new_col) not in bound: 
                    bound[(new_row, new_col)] = 0
    
    # This method takes in (i, j) position and check the presence of the pattern   
    # and how many there are around that position (horizontally, vertically and diagonally)
    def countPattern(self, i_0, j_0, pattern, score, bound, flag):
        '''
        pattern = key of patternDict --> tuple of patterns of various length
        score = value of patternDict --> associated score to pattern
        bound = dictionary with (i, j) as key and associated cell value as value
        flag = +1 if want to add the score, -1 if want to remove the score from the bound
        '''
        # Set unit directions
        directions = [(1, 0), (1, 1), (0, 1), (-1, 1)]
        # Prepare column, row, length, count
        length = len(pattern)
        count = 0

        # Loop through all 4 directions
        for dir in directions:
            # Find number of squares (max 5) that we can go back in each direction 
            # to check for the pattern indicated as parameter
            if dir[0] * dir[1] == 0:
                steps_back = dir[0] * min(5, j_0) + dir[1] * min(5, i_0)
            elif dir[0] == 1:
                steps_back = min(5, j_0, i_0)
            else:
                steps_back = min(5, N-1-j_0, i_0)
            # Very first starting point after finding out number of steps to go back
            i_start = i_0 - steps_back * dir[1]
            j_start = j_0 - steps_back * dir[0]

            # Move through all possible patterns in a row/col/diag
            z = 0
            while z <= steps_back:
                # Get a new starting point
                i_new = i_start + z*dir[1]
                j_new = j_start + z*dir[0]
                index = 0
                # Create a list storing empty positions that are fitted in a pattern
                remember = []
                # See if every square in a checked row/col/diag has the same status to a pattern
                while index < length and self.isValid(i_new, j_new, state=False) \
                        and self.boardMap[i_new][j_new] == pattern[index]: 
                    if self.isValid(i_new, j_new):
                        remember.append((i_new, j_new)) 
                    
                    i_new = i_new + dir[1]
                    j_new = j_new + dir[0]
                    index += 1

                # If we found one pattern
                if index == length:
                    count += 1
                    for pos in remember:
                        # Check whether pos is already present in bound dict
                        if pos not in bound:
                            bound[pos] = 0
                        bound[pos] += flag*score  # Update better percentage later in evaluate()
                    z += index
                else:
                    z += 1

        return count
    
    # This method takes in current board's value and intended move, and returns the value after that move is made
    # The idea of this method is to calculate the difference in number of patterns, thus value, 
    # around checked position, then add that difference to current board's value
    def evaluate(self, new_i, new_j, board_value, turn, bound):
        '''
        board_value = value of the board updated at each minimax and initialized as 0 
        turn = [1, -1] AI or human turn
        bound = dict of empty playable cells with corresponding score
        '''
        # Check cache first
        cache_key = (new_i, new_j, turn, self.rollingHash)
        if cache_key in self.evalCache:
            return self.evalCache[cache_key]
        
        value_before = 0
        value_after = 0
        
        # Check for immediate win/loss - highest priority
        self.boardMap[new_i][new_j] = turn
        if self.isFive(new_i, new_j, turn):
            self.boardMap[new_i][new_j] = 0
            result = board_value + (10000000 * turn)
            self.evalCache[cache_key] = result
            return result
        self.boardMap[new_i][new_j] = 0
        
        # Check for every pattern in patternDict
        for pattern in self.patternDict:
            score = self.patternDict[pattern]
            # For every pattern, count have many there are for new_i and new_j
            # and multiply them by the corresponding score
            value_before += self.countPattern(new_i, new_j, pattern, abs(score), bound, -1)*score
            # Make the move then calculate value_after
            self.boardMap[new_i][new_j] = turn
            value_after += self.countPattern(new_i, new_j, pattern, abs(score), bound, 1) *score
            
            # Delete the move
            self.boardMap[new_i][new_j] = 0

        result = board_value + value_after - value_before
        
        # Cache the result
        if len(self.evalCache) < 10000:  # Limit cache size
            self.evalCache[cache_key] = result
        
        return result

    ### MiniMax algorithm with AlphaBeta Pruning ###
    def alphaBetaPruning(self, depth, board_value, bound, alpha, beta, maximizingPlayer):

        if depth <= 0 or (self.checkResult() != None):
            return board_value # Static evaluation
        
        # Transposition table of the format {hash: [score, depth, flag]}
        if self.rollingHash in self.TTable and self.TTable[self.rollingHash][1] >= depth:
            return self.TTable[self.rollingHash][0] #return board value stored in TTable
        
        # Check for immediate threats - if found, narrow search
        threat_moves = self.detectThreats(bound, maximizingPlayer)
        if threat_moves:
            # Only search threat moves (forced moves)
            search_bound = {move: bound.get(move, 0) for move in threat_moves}
        else:
            # Limit search width for deeper nodes - only consider top moves
            if depth < self.depth and len(bound) > 10:
                # Sort and take top 10 moves
                sorted_moves = sorted(bound.items(), key=lambda x: x[1], reverse=True)[:10]
                search_bound = dict(sorted_moves)
            else:
                search_bound = bound
        
        # AI is the maximizing player 
        if maximizingPlayer:
            # Initializing max value
            max_val = -math.inf

            # Look through the all possible child nodes
            for child in self.childNodes(search_bound, depth):
                i, j = child[0], child[1]
                # Create a new bound with updated values
                # and evaluate the position if making the move
                new_bound = dict(search_bound)
                new_val = self.evaluate(i, j, board_value, 1, new_bound)
                
                # Make the move and update zobrist hash
                self.boardMap[i][j] = 1
                self.rollingHash ^= self.zobristTable[i][j][0] # index 0 for AI moves

                # Update bound based on the new move (i,j)
                self.updateBound(i, j, new_bound) 

                # Evaluate position going now at depth-1 and it's the opponent's turn
                eval = self.alphaBetaPruning(depth-1, new_val, new_bound, alpha, beta, False)
                if eval > max_val:
                    max_val = eval
                    if depth == self.depth: 
                        self.currentI = i
                        self.currentJ = j
                        self.boardValue = eval
                        self.nextBound = new_bound
                alpha = max(alpha, eval)

                # Undo the move and update again zobrist hashing
                self.boardMap[i][j] = 0 
                self.rollingHash ^= self.zobristTable[i][j][0]
                
                del new_bound
                if beta <= alpha: # prune
                    # Store as killer move
                    if depth < len(self.killerMoves):
                        if (i, j) not in self.killerMoves[depth]:
                            self.killerMoves[depth].append((i, j))
                            if len(self.killerMoves[depth]) > 3:  # Keep only top 3 killer moves
                                self.killerMoves[depth].pop(0)
                    break

            # Update Transposition Table
            utils.update_TTable(self.TTable, self.rollingHash, max_val, depth)

            return max_val

        else:
            # Initializing min value
            min_val = math.inf
            # Look through the all possible child nodes
            for child in self.childNodes(search_bound, depth):
                i, j = child[0], child[1]
                # Create a new bound with updated values
                # and evaluate the position if making the move
                new_bound = dict(search_bound)
                new_val = self.evaluate(i, j, board_value, -1, new_bound)

                # Make the move and update zobrist hash
                self.boardMap[i][j] = -1 
                self.rollingHash ^= self.zobristTable[i][j][1] # index 1 for human moves

                # Update bound based on the new move (i,j)
                self.updateBound(i, j, new_bound)

                # Evaluate position going now at depth-1 and it's the opponent's turn
                eval = self.alphaBetaPruning(depth-1, new_val, new_bound, alpha, beta, True)
                if eval < min_val:
                    min_val = eval
                    if depth == self.depth: 
                        self.currentI = i 
                        self.currentJ = j
                        self.boardValue = eval 
                        self.nextBound = new_bound
                beta = min(beta, eval)
                
                # Undo the move and update again zobrist hashing
                self.boardMap[i][j] = 0 
                self.rollingHash ^= self.zobristTable[i][j][1]

                del new_bound
                if beta <= alpha: # prune
                    # Store as killer move
                    if depth < len(self.killerMoves):
                        if (i, j) not in self.killerMoves[depth]:
                            self.killerMoves[depth].append((i, j))
                            if len(self.killerMoves[depth]) > 3:  # Keep only top 3 killer moves
                                self.killerMoves[depth].pop(0)
                    break

            # Update Transposition Table
            utils.update_TTable(self.TTable, self.rollingHash, min_val, depth)

            return min_val
    
    def detectThreats(self, bound, is_ai_turn):
        """
        Detect immediate threats (win in 1 move or must block)
        Returns list of critical moves that must be considered
        """
        threats = []
        player = 1 if is_ai_turn else -1
        opponent = -player
        
        for (i, j) in bound.keys():
            # Check if this move wins for current player
            self.boardMap[i][j] = player
            if self.isFive(i, j, player):
                self.boardMap[i][j] = 0
                return [(i, j)]  # Winning move - only consider this
            self.boardMap[i][j] = 0
            
            # Check if opponent can win here (must block)
            self.boardMap[i][j] = opponent
            if self.isFive(i, j, opponent):
                threats.append((i, j))
            self.boardMap[i][j] = 0
        
        return threats

    # Set the first move of the AI in (7,7) the center of the board
    def firstMove(self):
        self.currentI, self.currentJ = 7,7
        self.setState(self.currentI, self.currentJ, 1)

    # Check whether the game has ended and returns the winner if there is
    # otherwise, if there are no empty cells left, it's tie
    def checkResult(self):
        if self.isFive(self.currentI, self.currentJ, self.lastPlayed) \
            and self.lastPlayed in (-1, 1):
            return self.lastPlayed
        elif self.emptyCells <= 0:
            # tie
            return 0
        else:
            return None
    
    def getWinner(self):
        if self.checkResult() == 1:
            return 'Gomoku AI! '
        if self.checkResult() == -1:
            return 'Human! '
        else:
            return 'None'
    
    def clearCaches(self):
        """Clear evaluation cache to free memory"""
        self.evalCache.clear()
        if hasattr(self, 'cnn_cache'):
            self.cnn_cache.clear()
        # Keep transposition table as it's more valuable
        # self.TTable.clear()
    
    def resetGame(self):
        """Reset AI state for a new game"""
        self.boardMap = [[0 for j in range(N)] for i in range(N)]
        self.currentI = -1
        self.currentJ = -1
        self.nextBound = {}
        self.boardValue = 0
        self.turn = 0
        self.lastPlayed = 0
        self.emptyCells = N * N
        self.rollingHash = 0
        self.TTable.clear()
        self.evalCache.clear()
        self.killerMoves = [[] for _ in range(self.depth + 5)]
        if hasattr(self, 'cnn_cache'):
            self.cnn_cache.clear()
        print("✓ AI reset - Ready for new game!")

class CNN_GomokuAI(GomokuAI):
    def __init__(self, model_path="checkpoints/caro_best.pt", use_hybrid=True):
        super().__init__(depth=4)  # Shallow depth since CNN handles evaluation
        print("🧠 Loading CNN model for board evaluation...")
        self.use_hybrid = use_hybrid
        try:
            load_model_into_cache(model_path, use_fp16=True, use_ema=True)
            self.model_ready = True
            print("✅ CNN model loaded successfully!")
        except Exception as e:
            print(f"⚠️  Cannot load CNN model: {e}")
            print("⚠️  Falling back to heuristic-only evaluation")
            self.model_ready = False
        
        # CNN evaluation cache
        self.cnn_cache = {}

    def evaluate(self, new_i, new_j, board_value, turn, bound):
        """
        Hybrid evaluation: CNN + Heuristic patterns
        CNN provides strategic understanding, patterns provide tactical knowledge
        """
        if not self.model_ready or not self.use_hybrid:
            # Fallback to parent's heuristic evaluation
            return super().evaluate(new_i, new_j, board_value, turn, bound)
        
        # Check for immediate win/loss first (highest priority)
        self.boardMap[new_i][new_j] = turn
        if self.isFive(new_i, new_j, turn):
            self.boardMap[new_i][new_j] = 0
            return board_value + (10000000 * turn)
        self.boardMap[new_i][new_j] = 0
        
        # Get CNN evaluation
        import numpy as np
        cache_key = (new_i, new_j, turn, self.rollingHash)
        
        if cache_key in self.cnn_cache:
            cnn_score = self.cnn_cache[cache_key]
        else:
            board_copy = np.array(self.boardMap, dtype=np.float32)
            board_copy[new_i][new_j] = turn
            
            try:
                with torch.no_grad():
                    cnn_score = evaluate_model(board_copy, current_player=turn)
                    cnn_score = cnn_score.item() if hasattr(cnn_score, "item") else float(cnn_score)
                    # Scale CNN score to match heuristic range
                    cnn_score = cnn_score * 50000
            except Exception as e:
                cnn_score = 0.0
            
            # Cache result
            if len(self.cnn_cache) < 5000:
                self.cnn_cache[cache_key] = cnn_score
        
        # Get heuristic evaluation from parent
        heuristic_score = super().evaluate(new_i, new_j, board_value, turn, bound)
        
        # Hybrid: 60% heuristic (tactical), 40% CNN (strategic)
        return 0.6 * heuristic_score + 0.4 * cnn_score

    def get_candidate_moves(self, board, max_distance=2):
        """
        Generate candidate moves near existing stones
        More efficient than checking all empty cells
        """
        size = len(board)
        candidates = set()
        
        # Find all occupied cells
        occupied = []
        for i in range(size):
            for j in range(size):
                if board[i][j] != 0:
                    occupied.append((i, j))
        
        # If board is empty, return center
        if not occupied:
            return [(size//2, size//2)]
        
        # Add empty cells near occupied cells
        for i, j in occupied:
            for di in range(-max_distance, max_distance + 1):
                for dj in range(-max_distance, max_distance + 1):
                    ni, nj = i + di, j + dj
                    if 0 <= ni < size and 0 <= nj < size and board[ni][nj] == 0:
                        candidates.add((ni, nj))
        
        return list(candidates)

    def detectThreats(self, bound, is_ai_turn):
        """
        Enhanced threat detection using both pattern matching and CNN
        """
        threats = []
        player = 1 if is_ai_turn else -1
        opponent = -player
        
        # First check for immediate wins/blocks
        for (i, j) in bound.keys():
            # Check if this move wins for current player
            self.boardMap[i][j] = player
            if self.isFive(i, j, player):
                self.boardMap[i][j] = 0
                return [(i, j)]  # Winning move - only consider this
            self.boardMap[i][j] = 0
            
            # Check if opponent can win here (must block)
            self.boardMap[i][j] = opponent
            if self.isFive(i, j, opponent):
                threats.append((i, j))
            self.boardMap[i][j] = 0
        
        # If threats found, return them
        if threats:
            return threats
        
        # Check for critical patterns (live-4, double-3, etc.)
        critical_moves = []
        for (i, j) in list(bound.keys())[:15]:  # Check top 15 moves
            score = bound.get((i, j), 0)
            # High score indicates critical pattern
            if abs(score) > 50000:
                critical_moves.append((i, j))
        
        return critical_moves if critical_moves else []
