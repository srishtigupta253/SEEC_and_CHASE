#This code helps break the entire play script into multiple conversations based on the change in emotion
j=0
countall = 0

#One can either combine all plays and store them as allconvo or use a play one-by-one
for i in allconvo:
    if len(i) == 3:
        j=i[2] #each entry in the considered dataframe has the utterance, its emotion, and sentiment values across positive, negative, and neutral quantities
        if df['emotion'][j] == df['emotion'][j-1]:
            curremo = df['emotion'][j]
            if abs(df[curremo][j]-df[curremo][j-1]) > 28 or df['neu'][j-1]-df['neu'][j] > 0.28:
                for j in i:
                    print(df['actual text'][j])
                countall = countall+1
                print()
            else:
                continue
        else:
            for j in i:
                print(df['actual text'][j])
            countall = countall+1
            print()
    
    elif len(i) == 4:
#         print(i)
        for j in [i[2],i[3]]:
#             print(j)
            if df['emotion'][j] == df['emotion'][j-1]:
                curremo = df['emotion'][j]
                if abs(df[curremo][j]-df[curremo][j-1]) > 28 or df['neu'][j-1]-df['neu'][j] > 0.28:
                    for k in range(i[0],j+1):
                        print(df['actual text'][k])
                    countall = countall+1
                    print()
                    break
                else:
                    continue
            else:
                for k in range(i[0],j+1):
                    print(df['actual text'][k])                
                countall = countall+1
                print()
                break
                
    elif len(i) == 5:
#         print(i)
        for j in [i[2],i[3],i[4]]:
#             print(j)
            if df['emotion'][j] == df['emotion'][j-1]:
                curremo = df['emotion'][j]
                if abs(df[curremo][j]-df[curremo][j-1]) > 28 or df['neu'][j-1]-df['neu'][j] > 0.28:
                    for k in range(i[0],j+1):
                        print(df['actual text'][k])
                    countall = countall+1
                    print()
                    break
                else:
                    continue
            else:
                for k in range(i[0],j+1):
                    print(df['actual text'][k])                
                countall = countall+1
                print()
                break
    else:
        end = i[0]
        for j in i:
            if j==end or j==end+1:
                continue
            else:
                if df['emotion'][j] == df['emotion'][j-1]:
                    if df['neu'][j] == 1:
                        continue
                    else:
                        curremo = df['emotion'][j]
                        if abs(df[curremo][j]-df[curremo][j-1]) > 30 or df['neu'][j-1]-df['neu'][j] > 0.20:
    #                         print(end)
                            for k in range(end,j+1):
                                print(df['actual text'][k])
                            end = j+1
                            countall = countall+1
                            print()
                        else:
                            continue
                else:
                    curremo = df['emotion'][j]
                    if df[curremo][j] > 75:
                        for k in range(end,j+1):
                            print(df['actual text'][k])
                        end = j+1
                        countall = countall+1
                        print()
                    else:
                        continue
