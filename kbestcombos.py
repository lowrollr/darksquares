
import heapq


def s_best_k_combos(nums, k, s):
    nums.sort(key=lambda x: -x)
    solution = []
    count = 0
    first = (-sum([nums[i] for i in range(k)]), count, tuple([i for i in range(k)]))
    heap = [first]
    combos = set()
    while len(solution) < s and heap:
        print([([nums[i] for i in c], s) for s,_,c in heap])
        best_sum, _, combo = heapq.heappop(heap)
        best_sum = -best_sum
        solution.append([nums[i] for i in combo])
        
        # relax pointers
        for j in range(k-1):
            if combo[j] + 1 != combo[j+1]:
                new_combo = list(combo)
                new_combo[j] += 1
                old_value = nums[combo[j]]
                new_value = nums[combo[j] + 1]
                new_sum = best_sum - (old_value - new_value)
                new_combo = tuple(new_combo)
                if new_combo not in combos:
                    combos.add(new_combo)
                    heapq.heappush(heap, (-new_sum, count, tuple(new_combo)))
                    count += 1
                    if len(heap) > s:
                        heapq.heappop(heap)

        # relax last pointer (check that its not out of bounds)
        if combo[-1] + 1 < len(nums):
            new_combo = list(combo)
            old_value = nums[combo[-1]]
            new_value = nums[combo[-1] + 1]
            new_combo[-1] += 1
            
            new_sum = best_sum - (old_value - new_value)
            new_combo = tuple(new_combo)
            if new_combo not in combos:
                combos.add(new_combo)
                heapq.heappush(heap, (-new_sum, count, new_combo))
                count += 1
                if len(heap) > s:
                    heapq.heappop(heap)
    
    return solution



print(s_best_k_combos([8,6,2,10,9,11,20,21,27,30,99], 10, 2000))