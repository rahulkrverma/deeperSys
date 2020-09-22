

import os
import json


def read_mainJson(filaPath):

	manager = {}
	watcher = {}
	temp = {}

	dump = open(filaPath, 'rb')
	dump = json.load(dump)
	for x in dump:
		per_ = x['priority']
		name_ = x['name']
		temp[per_] = name_

	for _ in dump:
		managerIn = _['managers']
		watcherIn = _['watchers']
		per = _["priority"]
		for m in managerIn:
			if m in manager:
				te = manager[m]
				te.append(per)
				manager[m] = te
			else:
				manager[m] = [per]

		for w in watcherIn:
			if w in watcher:
				te_ = watcher[w]
				te_.append(per)
				watcher[w] = te_
			else:
				watcher[w] = [per]

	for x in manager:
		te = manager[x]
		te.sort()
		new = []
		for i in te:
			new.append(temp[i])
		manager[x] = new

	for x in watcher:
		te = watcher[x]
		te.sort()
		new = []
		for i in te:
			new.append(temp[i])
		watcher[x] = new

	return manager, watcher


if __name__ == '__main__':
	
	manager, watcher = read_mainJson(filaPath = 'source_file_2.json')

	with open("managers.json", "w") as outfile: 
		json.dump(manager, outfile)

	with open("watchers.json", "w") as outfile: 
		json.dump(watcher, outfile)


