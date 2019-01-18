import json

with open("sankey_Argentina.json",'r') as w:
    data = json.load(w)
node = []
link = []
for i in range(len(data) - 1):
    if data[i]['next'] == 1:
        this_formation = data[i]['formation'].copy()
        next_formation = data[i + 1]['formation'].copy()
        # source = data[i]
        # target = data[i + 1]
        for j in range(1,4):
            name = str(data[i]['frame']) + '-' + str(j)
            node.append({"name":name})
            for k in range(1,4):
                value = 0
                for l in this_formation[j - 1]:
                    if l in next_formation[k - 1]:
                        value += 1
                if value > 0:
                    print({"source":name,"target":str(data[i + 1]['frame']) + '-' + str(k),"value":value})
                    link.append({"source":name,"target":str(data[i + 1]['frame']) + '-' + str(k),"value":value})
            if data[i + 1]['next'] == 0:
                name = str(data[i + 1]['frame']) + '-' + str(j)
                node.append({"name":name})
# print(node)
# print(link)

final = {"nodes":node,"links":link}
with open("node-link_Argentina.json",'w+') as w:
    json.dump(final,w,indent = 4)