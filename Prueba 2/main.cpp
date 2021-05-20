#include "opencv2/objdetect.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/videoio.hpp"
#include <iostream>
using namespace std;
using namespace cv;

void detectAndDisplay(Mat frame);

//Instanciamos dos clasificadores
CascadeClassifier face_cascade;
CascadeClassifier eyes_cascade;

int main(int argc, const char** argv)
{
    CommandLineParser parser(argc, argv,
        "{help h||}"
        "{face_cascade|data/haarcascades/haarcascade_frontalface_alt.xml|Path to face cascade.}"
        "{eyes_cascade|data/haarcascades/haarcascade_eye_tree_eyeglasses.xml|Path to eyes cascade.}"
        "{camera|0|Camera device number.}");
    parser.about("\nEste programa es una demostracion sobre la deteccion de objetos con clasificadores admitida por OpenCV\n\n");
    parser.printMessage();

    //Buscamos que esten los archivos
    String face_cascade_path = samples::findFile(parser.get<String>("face_cascade"));
    String eyes_cascade_path = samples::findFile(parser.get<String>("eyes_cascade"));


    //Cargamos los clasificadores
    if (!face_cascade.load(face_cascade_path))
    {
        cout << "Error cargando el clasificador de caras\n";
        return -1;
    };
    if (!eyes_cascade.load(eyes_cascade_path))
    {
        cout << "Error cargando el clasificador de ojos\n";
        return -1;
    };


    int camera_device = parser.get<int>("camera");
    VideoCapture capture;
    //Abrimos la camaraque paso el usuario
    capture.open(camera_device);
    if (!capture.isOpened())
    {
        cout << "Error abriendo la camara\n";
        return -1;
    }
    Mat frame;
    while (capture.read(frame))
    {
        if (frame.empty())
        {
            cout << "No se puede capturar el frame de la camara\n";
            break;
        }
        //Detectar y mostrar
        detectAndDisplay(frame);
        if (waitKey(10) == 27)
        {
            break; // escape
        }
    }
    return 0;
}
void detectAndDisplay(Mat image)
{
    //Adaptamos el frame de video para que este en escala de grises
    Mat frame_gray;
    cvtColor(image, frame_gray, COLOR_BGR2GRAY);
    //Ecualizamos el histograma (Tomamos el histograma y los distribuimos mas para generar un mayor contraste en la imagen : https://docs.opencv.org/3.4/d4/d1b/tutorial_histogram_equalization.html)
    equalizeHist(frame_gray, frame_gray);
    std::vector<Rect> faces;    //Vector donde se guardan las bounding boxes para las caras encontradas
    //LINEA CLAVE, aqui se le pasa al clasificador la imagen y el vector para que nos devuelva las coincidencias
    face_cascade.detectMultiScale(frame_gray, faces);      

    for (size_t i = 0; i < faces.size(); i++)
    {
        //Dibujamos un rectangulo alrededor de la cara
        rectangle(image, faces[i], Scalar(255, 0, 255), 4);

        //Recortamos la cara de la imagen para ver los ojos
        Mat faceROI = frame_gray(faces[i]);                 

        //Mismo procedimiento que con la cara
        std::vector<Rect> eyes;
        eyes_cascade.detectMultiScale(faceROI, eyes);

        for (size_t j = 0; j < eyes.size(); j++)
        {
            //Dibuajmos un rectangulo en cada ojo
            eyes[j].x += faces[i].x;
            eyes[j].y += faces[i].y;
            rectangle(image, eyes[j], Scalar(255, 0, 0), 4);
        }
    }

    //Mostramos el frame procesado
    imshow("Deteccion de rostro y ojos", image);
}