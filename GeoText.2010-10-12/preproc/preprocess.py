train_file_name = "full_text.txt"

f = open(train_file_name, "r", encoding="ISO-8859-1")
lines = f.readlines()
print(lines[0])


for line in lines:
  fields = line.split()
  if len(fields) != 4:
      pass