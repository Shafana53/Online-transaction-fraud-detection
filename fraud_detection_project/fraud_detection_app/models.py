from django.db import models

class Transaction(models.Model):

    type = models.CharField(max_length=255)
    amount = models.FloatField()
    nameOrig = models.CharField(max_length=255)
    oldbalanceOrg = models.FloatField()
    newbalanceOrig = models.FloatField()
    nameDest = models.CharField(max_length=255)
    oldbalanceDest = models.FloatField()
    newbalanceDest = models.FloatField()
    isFraud = models.BooleanField(default=False)








