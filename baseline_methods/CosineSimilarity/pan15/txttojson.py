import json
  
  
# the file to be converted
filename = 'test/test-dataset/truth.txt'
  
# resultant dictionary
dict1 = []
  
# fields in the sample file 
fields =['name','answer']
  
with open(filename) as fh:
      
  
      
    # count variable for employee id creation
    l = 1
    for line in fh:
          
        # reading line by line from the text file
        description = list( line.strip().split(None, 4))
          
        # for output see below
        print(description) 
      
        # loop variable
        i = 0
        # intermediate dictionary
        dict2 = {}
        while i<len(fields):
              
                # creating dictionary for each employee
                dict2[fields[i]]= description[i]
                i = i + 1
                  
        # appending the record of each employee to
        # the main dictionary
        dict1.append(dict2)
        l = l + 1
  
with open("test/test-dataset/truth.json", mode='w') as writer:
    json.dump(dict1, writer, ensure_ascii=False)
