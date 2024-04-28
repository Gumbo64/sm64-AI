import numpy as np
def calc_spread(nums,max_num=False):
    if not max_num:
        return np.log(np.mean(nums)) - np.mean(np.log(nums))
    return np.log(max_num) - np.mean(np.log(nums))

def print_spread(nums,max_num=False):
    print(f"{nums}: {calc_spread(nums,max_num=max_num)}")

print_spread([1,2,3])
print_spread([1,1,2,2,3,3])
print_spread([10,20,30])
print_spread([1,2,3,4])

# nums = [1,2,3,4]
# weights = [0.2,0.5,0.1,0.2]
# result = [nums[i] * weights[i] for i in range(len(nums))]
print_spread([1,2,3,4],max_num=4)
print_spread([1,1,1,1],max_num=4)

