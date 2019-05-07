import json
f = open('test.json', 'r')
content = f.read()
jContent = json.loads(content)
# print(jContent)
xmin = 100
xmax = 0
ymin = 100
ymax = 0
newJson = '{"map": ['
for nod in jContent['NOD']:
    x, y = nod['COORDINATES']
    newJson += '{"name": "' + nod["NAME"] + '", "x": %.10f, "y": %.10f},' % (x, y)
    if xmin > x:
        xmin = x
    if xmax < x:
        xmax = x
    if ymax < y:
        ymax = y
    if ymin > y:
        ymin = y
newJson = newJson[:-1]
newJson += ']}'
print(json.loads(newJson))
jsontype = json.loads(newJson)

# print('x: min', xmin, 'max', xmax)
# print('y: min', ymin, 'max', ymax)

font = cv2.FONT_HERSHEY_SIMPLEX
cv2.putText(img, 'OpenCV', (10, 500), font, 4, (255, 255, 255), 2, cv2.LINE_AA)


