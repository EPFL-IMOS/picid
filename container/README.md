# Tips & Tricks

If you want to use WandB, deploy your secret:
```bash
$ kubectl create secret generic wandb-secret --from-literal secret1=<secret>
secret/wandb-secret created
```