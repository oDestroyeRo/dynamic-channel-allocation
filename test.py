list_code = []
channels = 10
count = 0
col = channels
for j in range(col):
    num_set =  channels - col

    for k in range(channels-num_set):
        array = []
        
        for l in range(num_set + 1):
            array.append(l+k)
            print(l+k, end='')
        list_code.append(array)
        print()
        count += 1

    col-=1
print(count)

print(list_code[54])





temp = 1
channel_num = 0
for i in range(channels):
    channel_num += temp
    temp += 1

print(channel_num)