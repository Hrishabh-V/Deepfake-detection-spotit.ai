from django.db import models

# Create your models here.

class Login(models.Model):
    username=models.CharField(max_length=100)
    password=models.CharField(max_length=100)
    type=models.CharField(max_length=100)

class User(models.Model):
    name=models.CharField(max_length=100)
    email=models.CharField(max_length=100)
    phone_number=models.CharField(max_length=100)
    LOGIN=models.ForeignKey(Login,on_delete=models.CASCADE)

class file_upload(models.Model):
    image=models.CharField(max_length=500)
    video=models.CharField(max_length=500)

