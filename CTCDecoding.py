import numpy as np


def clean_path(path):
	""" utility function that performs basic text cleaning on path """

	# No need to modify
	path = str(path).replace("'","")
	path = path.replace(",","")
	path = path.replace(" ","")
	path = path.replace("[","")
	path = path.replace("]","")

	return path


class GreedySearchDecoder(object):

    def __init__(self, symbol_set):
        """
        
        Initialize instance variables

        Argument(s)
        -----------

        symbol_set [list[str]]:
            all the symbols (the vocabulary without blank)

        """

        self.symbol_set = symbol_set
        print(symbol_set)

    def decode(self, y_probs):
        """

        Perform greedy search decoding

        Input
        -----

        y_probs [np.array, dim=(len(symbols) + 1, seq_length, batch_size)]
            batch size for part 1 will remain 1, but if you plan to use your
            implementation for part 2 you need to incorporate batch_size

        Returns
        -------

        decoded_path [str]:
            compressed symbol sequence i.e. without blanks or repeated symbols

        path_prob [float]:
            forward probability of the greedy path

        """

        decoded_path = []
        blank = 0
        path_prob = 1

        # TODO:
        # 1. Iterate over sequence length - len(y_probs[0])
        # 2. Iterate over symbol probabilities
        # 3. update path probability, by multiplying with the current max probability
        # 4. Select most probable symbol and append to decoded_path
    
    
        for i in range(len(y_probs[0])):
            for j in range(y_probs.shape[0]):
                max_idx = np.argmax(y_probs[:,i,:])
                
            path_prob *= max(y_probs[:,i,:])           
            max_sym = self.symbol_set[max_idx-1]
            decoded_path.append(max_sym)
            
        compressed_path = [decoded_path[i] for i in range(len(decoded_path)) if i < (len(decoded_path)-1) and decoded_path[i] != decoded_path[i+1]]
        if compressed_path[-1] == decoded_path[-1]:
            pass
        else:
            compressed_path.append(decoded_path[-1])
            
        decoded_path = compressed_path
            
            

        decoded_path = clean_path(decoded_path)

        return decoded_path, path_prob


class BeamSearchDecoder(object):

    def __init__(self, symbol_set, beam_width):
        """

        Initialize instance variables

        Argument(s)
        -----------

        symbol_set [list[str]]:
            all the symbols (the vocabulary without blank)

        beam_width [int]:
            beam width for selecting top-k hypotheses for expansion

        """

        self.symbol_set = symbol_set
        self.beam_width = beam_width
        self.pathscore, self.blank_path_score = {},{}

    def decode(self, y_probs):
        """
        
        Perform beam search decoding

        Input
        -----

        y_probs [np.array, dim=(len(symbols) + 1, seq_length, batch_size)]
 			batch size for part 1 will remain 1, but if you plan to use your
 			implementation for part 2 you need to incorporate batch_size

        Returns
        -------
        
        forward_path [str]:
            the symbol sequence with the best path score (forward probability)

        merged_path_scores [dict]:
            all the final merged paths with their scores

        """
    
        
    

        # TODO:
        # 1. Iterate over sequence length - len(y_probs[0])
        #    - initialize a list to store all candidates
        # 2. Iterate over 'sequences'
        # 3. Iterate over symbol probabilities
        #    - Update all candidates by appropriately compressing sequences
        #    - Handle cases when current sequence is empty vs. when not empty
        # 4. Sort all candidates based on score (descending), and rewrite 'ordered'
        # 5. Update 'sequences' with first self.beam_width candidates from 'ordered'
        # 6. Merge paths in 'ordered', and get merged paths scores
        # 7. Select best path based on merged path scores, and return
        
        new_paths_terminal_blank, new_paths_terminal_symbol, new_blank_path_score, new_path_score = self.initial_path(self.symbol_set, y_probs[:,0])
        
        
        for t in range(1,len(y_probs[0])):
            # print("t",t)
            paths_terminal_blank, paths_terminal_symbol, self.blank_path_score, self.pathscore = self.prune(new_paths_terminal_blank, new_paths_terminal_symbol, new_blank_path_score, new_path_score, self.beam_width)
            # print("pruned_paths_terminal_blank",paths_terminal_blank)
            # print("pruned_paths_terminal_symbol",paths_terminal_symbol)
            # print("path score",self.pathscore)
        
            
            new_paths_terminal_blank, new_blank_path_score = self.extend_with_blank(paths_terminal_blank,paths_terminal_symbol,y_probs[:,t])
            
            new_paths_terminal_symbol, new_path_score = self.extend_with_symbol(paths_terminal_blank,paths_terminal_symbol, self.symbol_set, y_probs[:,t])
            
            
        merged_paths, final_path_score = self.merge_identical_paths(new_paths_terminal_blank,new_blank_path_score,new_paths_terminal_symbol,new_path_score)
        
        # print("merged paths", merged_paths)
            
            
            #best_path = np.argmax(final_path_score)
        
        final_path_score = {k: v for k, v in sorted(final_path_score.items(), key=lambda item: item[1], reverse=True)}
        for key in final_path_score.keys():
            final_path_score[key] = final_path_score[key][0]
        
        final_score_path = {y:x for x, y in final_path_score.items()}
        
        
        best_path = final_score_path[max(final_score_path.keys())]
        
        # print("check", check)
        # print("best path", best_path)
            
        # print("paths terminal blank",paths_terminal_blank)
        # print("paths terminal symbol", paths_terminal_symbol)
        # print("blank path score", blank_path_score)
        # print("pruned_path_score")

        merged_path_scores = final_path_score
        
        return best_path, merged_path_scores
    
        
                    
    def initial_path(self,symbol_set, y_probs):
        initial_blank_path_score, initial_path_score = {}, {}
        path = ""
        initial_blank_path_score[path] = y_probs[0]
        initial_paths_final_blank = [path]
        
        initial_path_with_final_symbol = set()
        for c in symbol_set:
            path = c
            symbol_idx = symbol_set.index(c)
            initial_path_score[path] = y_probs[symbol_idx+1]
            initial_path_with_final_symbol.add(path)
            
        return initial_paths_final_blank, initial_path_with_final_symbol, initial_blank_path_score, initial_path_score
    
    def prune(self,paths_terminal_blank, paths_terminal_symbol, blank_path_score,pathscore, beam_width):
        
        pruned_blank_path_score = {}
        pruned_path_score = {}
        scorelist = {}
        i = 1
        paths_terminal_symbol,paths_terminal_blank = list(paths_terminal_symbol), list(paths_terminal_blank)
        
        # print(paths_terminal_blank)
        # for  p in range(len(paths_terminal_blank)):
        #     print(blank_path_score[paths_terminal_blank[p]])
        # print("terminal blank",paths_terminal_blank)
        # print("terminal symbol",paths_terminal_symbol)
        # print("pathscore", pathscore)
        
        
        for p in range(len(paths_terminal_blank)):
            # print("i", i)
            scorelist[i] = blank_path_score[paths_terminal_blank[p]]
            i += 1
            # print("no error")
        for p in range(len(paths_terminal_symbol)):
            # print("i", i)
            scorelist[i] = pathscore[paths_terminal_symbol[p]]
            i += 1
            # print("no error")
            
        # initial = scorelist
        
            
        for i in range(len(scorelist.keys())):
            scorelist[i+1] = scorelist[i+1][0]
            
        
        
        
        scorelist = {k: v for k, v in sorted(scorelist.items(), key=lambda item: item[1], reverse= True)}
        # print(scorelist)
        cutoff = list(scorelist.items())[self.beam_width-1][1] if self.beam_width<len(scorelist) else list(scorelist.items())[self.beam_width][-1]
        # print(cutoff)
        
        
        pruned_paths_terminal_blank = set()
        for p in range(len(paths_terminal_blank)):
            if blank_path_score[paths_terminal_blank[p]] >= cutoff:
                pruned_paths_terminal_blank.add(paths_terminal_blank[p])
                pruned_blank_path_score[paths_terminal_blank[p]] = blank_path_score[paths_terminal_blank[p]]
        pruned_paths_terminal_symbol = set()
        for p in range(len(paths_terminal_symbol)):
            if pathscore[paths_terminal_symbol[p]] >= cutoff:
                pruned_paths_terminal_symbol.add(paths_terminal_symbol[p])
                
                pruned_path_score[paths_terminal_symbol[p]] = pathscore[paths_terminal_symbol[p]]
                
                
                
        return (pruned_paths_terminal_blank, pruned_paths_terminal_symbol, pruned_blank_path_score,pruned_path_score)
    
    
    
    
    def extend_with_blank(self,paths_terminal_blank, paths_terminal_symbol, y_probs):
        
        updated_paths_terminal_blank = set()
        updated_blank_path_score = {}
        
        for path in paths_terminal_blank:
            updated_paths_terminal_blank.add(path)
            
            
            updated_blank_path_score[path] = self.blank_path_score[path]*y_probs[0][0]
    
                
        for path in paths_terminal_symbol:
            if path in updated_paths_terminal_blank:
            
                updated_blank_path_score[path] += self.pathscore[path]*y_probs[0][0]
            else:
                updated_paths_terminal_blank.add(path)
                updated_blank_path_score[path] = self.pathscore[path]*y_probs[0][0]
                
        return (updated_paths_terminal_blank, updated_blank_path_score)
                
        
    
    
    
    def extend_with_symbol(self,paths_terminal_blank,paths_terminal_symbol,symbol_set,y_probs):
    
        updated_paths_terminal_symbol = set()
        updated_path_score = {}
        paths_terminal_blank = list(paths_terminal_blank)
        paths_terminal_symbol = list(paths_terminal_symbol)
        
        for path in paths_terminal_blank:
            for c in symbol_set:
                newpath = path + c
                symbol_idx = symbol_set.index(c)
                updated_paths_terminal_symbol.add(newpath)
                updated_path_score[newpath] = self.blank_path_score[path]*y_probs[symbol_idx+1]
            #     print("no error")
            # print("upper loop no error")
        
        # print("paths terminal", paths_terminal_symbol)
        for path in paths_terminal_symbol:
            for c in symbol_set:
                newpath = path if c == path[-1] else (path +c)
                if newpath in updated_paths_terminal_symbol:
                    symbol_idx = symbol_set.index(c)
                    updated_path_score[newpath] += self.pathscore[path] *y_probs[symbol_idx+1]
                    #print("if no error")
                else:
                    # print("else")
                    updated_paths_terminal_symbol.add(newpath)
                    symbol_idx = symbol_set.index(c)
                    updated_path_score[newpath] = self.pathscore[path]*y_probs[symbol_idx+1]
                    #print("else: no error")
                    
        return updated_paths_terminal_symbol, updated_path_score
                   
           
    def merge_identical_paths(self,paths_terminal_blank,blank_path_scores, paths_terminal_symbol, path_score):
        merged_paths = paths_terminal_symbol
        final_path_score = path_score
        # print("merged paths",merged_paths)
        # print("path score",final_path_score)
        # print("blank scores",blank_path_scores)
        
        for p in paths_terminal_blank:
            
            if p in merged_paths:
                final_path_score[p] += blank_path_scores[p]
                
            else:
                merged_paths.add(p)
                final_path_score[p] = blank_path_scores[p]
                
        return merged_paths, final_path_score
    
            
        

            
            
            
          
          
                    
            
        
        
        

        
    
