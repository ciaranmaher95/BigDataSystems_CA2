mongod –replSet bigdata_ca2 –dbpath C:\data\db –port 27011 –rest
mongod –replSet bigdata_ca2 –dbpath ./mongoca2_1 –port 27012 –rest
mongod –replSet bigdata_ca2 –dbpath ./mongoca2_2 –port 27013 –rest

rs.initiate({
	_id: 'bigdata_ca2',
	members: [
		{_id: 1, host: 'localhost:27011'},
		{_id: 2, host: 'localhost:27012'},
		{_id: 3, host: 'localhost:27013'}
	]
});


mongo localhost:27011
mongo localhost:27012
mongo localhost:27013