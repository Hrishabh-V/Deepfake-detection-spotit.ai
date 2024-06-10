from django.core.files.storage import FileSystemStorage
from django.http import HttpResponse
from django.shortcuts import render
from .test import imagepredict
# Create your views here.
from sp_app.models import Login, User, file_upload


def ab(request):
    return render(request,'index.html')


def abc(request):
    return render(request,'indexhm.html')
def addfile(request):
    return render(request,'indexhm.html')

from django.core.files.storage import FileSystemStorage
from django.http import HttpResponse
from django.shortcuts import render
from .test import imagepredict
from sp_app.models import Login, User, file_upload

from django.shortcuts import render
from django.http import HttpResponse
from django.core.files.storage import FileSystemStorage
from .test import imagepredict

def addfilepost(request):
    if request.method == 'POST':
        if 'file' in request.FILES:
            photo = request.FILES['file']
            # Process the file
            lists = []
            from datetime import datetime
            date = datetime.now().strftime('%Y%m%d-%H%M%S') + '.jpg'
            fs = FileSystemStorage()
            fn = fs.save(photo.name, photo)
            if fn.endswith(".mp4") or fn.endswith(".avi"):
                import cv2
                import numpy as np
                count = 0
                count1 = 0
                res = []
                i = 0
                cap = cv2.VideoCapture(r"C:\Users\SHAHEEM\PycharmProjects\sp\media/"+fn)
                if not cap.isOpened():
                    return HttpResponse("Error opening video file")

                while cap.isOpened():
                    ret, frame = cap.read()
                    if ret:
                        if count % 20 == 0:
                            count1 += 1
                            cv2.imwrite(r"C:\Users\SHAHEEM\PycharmProjects\sp\media\sample.png", frame)
                            res = imagepredict(r"C:\Users\SHAHEEM\PycharmProjects\sp\media\sample.png")
                            lists.append(res)
                    else:
                        break

                    count += 1
                    pc = 0
                    nc = 0
                    res = []
                    for i in lists:
                        if i[1] == "Deepfake":
                            nc += 1
                        else:
                            pc += 1
                    if nc > pc:
                        c = nc / (nc + pc)
                        res.append(c)
                        res.append("DeepFake")
                    else:
                        c = pc / (nc + pc)
                        res.append(c)
                        res.append("Real")
                    return render(request, "indexhm.html", {"val": res})

                cap.release()
                cv2.destroyAllWindows()
            res = imagepredict(r"C:\Users\SHAHEEM\PycharmProjects\sp\media/"+fn)
            return render(request, "indexhm.html", {"val": res})
        else:
            # JavaScript alert for no file uploaded
            return render(request, "indexhm.html", {"popup": True})
    else:
        # Handle GET request or other cases
        return HttpResponse("Invalid request method.")




def user_register(request):
    return render(request,'user_register.html')

def user_register_POST(request):
    name=request.POST['name']
    email = request.POST['email']
    phone = request.POST['phone']
    password = request.POST['password']
    play=Login()
    play.username=email
    play.password=password
    play.type='user'
    play.save()

    run=User()
    run.name=name
    run.email=email
    run.phone_number=phone
    run.LOGIN=play
    run.save()
    return HttpResponse('''<script>alert('Registration Successfull');window.location="/login"</script>''')


def login(request):
    return render(request,'login.html')

def login_post(request):
    username=request.POST['name']
    password=request.POST['password']
    play=Login.objects.filter(username=username,password=password)
    if play.exists():
        run=Login.objects.get(username=username,password=password)
        request.session['lid']=run.id
        if run.type == 'user':
            return HttpResponse('''<script>alert('Login Successfull');window.location="/addfile"</script>''')
        else:
            return HttpResponse('''<script>alert('Invalid  username or password');window.location="/login"</script>''')
    else:
        return HttpResponse('''<script>alert('Invalid  username or password');window.location="/login"</script>''')


