# Server_app
## clinet_manager.py
`threding.Conditon()`の使い方がおかしい気がする（排他制御になっていない）．本来の意図としては`self.clients: Dict[str, ClientProxy]`を共有資源として，Clientの登録や削除のオペレーションをしたいのでは．現状は↓
```python:client_manager.py=
        self.clients[client.cid] = client
        with self._cv:
            self._cv.notify_all()
```
正しくはこうでは？
```python:client_manager.py=
        with self._cv:
            self.clients[client.cid] = client
            self._cv.notify_all()
```
Client削除も同じ？ 