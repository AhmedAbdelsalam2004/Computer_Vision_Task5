from django.urls import path
from . import views

urlpatterns = [
    path('status', views.status, name='status'),
    path('eigenfaces', views.get_eigenfaces, name='eigenfaces'),
    path('roc', views.get_roc, name='roc'),
    path('preview', views.preview, name='preview'),
    path('recognize', views.recognize, name='recognize'),
]
