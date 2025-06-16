import * as fs from 'fs'
var newobj = {
    synsets: {}
}
var arr = {}

var obj = JSON.parse(fs.readFileSync('./attribute_synsets.json', 'utf8'));


for (var key in obj) {
    if (obj.hasOwnProperty(key)) {
        if (!newobj.synsets[obj[key]]) newobj.synsets[obj[key]] = []
        newobj.synsets[obj[key]].push(key)

    }
}
let i = 0;
for (var te in newobj.synsets) {
    arr[i] = {}
    arr[i]['_id'] = te;
    arr[i]['values'] = newobj.synsets[te];
    i++;
}

fs.writeFileSync('file.json', JSON.stringify(arr));