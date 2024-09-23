'''
1. 
you are given an array of houses in a neighboorhood in a city.
you have to rearrange houses in such a way that in a single neighbourhood the houses are sorted by number in ascending order and no 2 houses with same number are in same neighbourhood.
you can only rearrange house based on the capacity of each neighbourhood . If neighbourhood "1" in input has 2 houses then at output also it can only have 2 houses.
For example-
{
{1,2},
{4,4,7,8},
{4,9,9,9}
}
becomes
{
{4,9},
{1,2,4,9},
{4,7,8,9}
}

inputs: houses: List[List[str]]
througts:
1. traverse list, put in first heap: neighborhood = (capacity, index), maintain a dict to store results of each neiborhood: result = {0:[1,2,3]}
2. traverse each sublist, put in the second heap: street_numbers = (count, val), where count denotes number of house to be allocated

while True:
    count, st_num = heappop(street_numbers)
    tmp = []
    for i in range(count):
        capacity, neib = heappop(neighborhood)
        result[neib].append(st_num)
        tmp.append((capacity, neib))
    
    for item in tmp:
        capacity, neib = item[0], item[1]
        capacity -= 1
        if capacity > 0:
            heappush(neighborhood, (capacity, neib))
        
    #count -= 1
    #if count > 0:
    #    heappush(street_numbers, (count, st_num))

'''