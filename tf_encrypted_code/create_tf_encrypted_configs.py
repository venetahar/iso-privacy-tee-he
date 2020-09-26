from collections import OrderedDict
import tf_encrypted as tfe

players = OrderedDict([
        ('server0', 'localhost:4000'),
        ('server1', 'localhost:4001'),
        ('server2', 'localhost:4002'),
    ])

config = tfe.RemoteConfig(players)
config.save('/tmp/tfe.config')
