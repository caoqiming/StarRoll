#include <opencv2/opencv.hpp>
#include<vector>
#include<cmath>
#include <cstdlib> // Header file needed to use srand and rand
#include <ctime> // Header file needed to use time
#include <boost/asio/post.hpp>
#include <boost/asio/thread_pool.hpp>
#include <boost/bind/bind.hpp>
using namespace std;
using namespace cv;

#define blue_threshold 60
#define green_threshold 60
#define red_threshold 60
constexpr double angle_step = 0.002;
constexpr double pi = 3.14159;

inline double rand_percent(int min,int max){
    return static_cast<double>((rand() % (max+1)+min)/100.0);
}

struct point{
    int i;
    int j;
    point(int i,int j):i(i),j(j){}
};

class Mystar{
public:
    const double max_r = 1;
    const double r_skip = 20; // 忽略在max_r与r_skip之间的点
    int center_i,center_j,nums;
    double r,ave_i,ave_j;
    vector<point>points;
    Vec3b ave_color;

    Mystar(int i ,int j):center_i(i),center_j(j),r(0),nums(1){
        points.emplace_back(point(i, j));
        ave_i = i;
        ave_j = j;
    }

    inline double distance(int i, int j, int x, int y){
        return pow(pow(i-x,2)+pow(j-y,2), 0.5);
    }

    bool try_to_add(int i,int j,bool &flag_skip){
        double dis = distance(i, j, center_i, center_j);
        if ( dis > max_r){
            if(dis<r_skip)
                flag_skip = true;
            return false;//很近但不够进，当作加入了以免生成新的
        }
            
        r = (r * nums + dis) / (nums + 1);
        ave_i = (ave_i * nums + i) / (nums + 1);
        ave_j = (ave_j * nums + j) / (nums + 1);
        center_i = static_cast<int>(ave_i);
        center_j = static_cast<int>(ave_j);
        points.emplace_back(point(i, j));
        nums++;
        return true;
    }

    void calculate_r(){
        r = 0;
        for(const auto &it:points){
            r = max(r, distance(it.i, it.j, center_i, center_j));
        }
    }

    void calculate_color(Mat &img){
        ave_color=img.at<Vec3b>(center_i, center_j);
    }

    void draw_new_star(Mat &img_old, Mat &img ,int new_center_i,int new_center_j){
        int new_i, new_j;
        int i_max = img.rows;
        int j_max = img.cols;
        double flash = rand_percent(80, 100);
        for(const auto &it:points){
            new_i = it.i - center_i + new_center_i;
            new_j = it.j - center_j + new_center_j;
            if (new_i < 0 || new_i >= i_max)
                return;
            if (new_j < 0 || new_j >= j_max)
                return;
            Vec3b &color_old=img_old.at<Vec3b>(it.i, it.j);
            Vec3b &color=img.at<Vec3b>(new_i, new_j);
            for(int ii=0;ii<3;++ii){
                color[ii] = static_cast<uchar>(color_old[ii] * flash);
            }
        }
    }
};

void roll_one_star(Mystar* it, int roll_i,int roll_j ,double angle_abs,double angle_step_used,Mat &img,Mat &img_old,Mat &img_temp){
    int i_max = img_old.rows;
    int j_max = img_old.cols;
    int relative_i,relative_j;
    relative_i = it->center_i - roll_i;
    relative_j = it->center_j - roll_j;
    int new_center_i, new_center_j;
    for(double thet =0;abs(thet) < angle_abs;thet+=angle_step_used){
        new_center_i = relative_i * cos(thet) - relative_j*sin(thet) + roll_i;
        new_center_j = relative_i * sin(thet) + relative_j*cos(thet) + roll_j;
        if (new_center_i < 0 || new_center_i >= i_max)
            continue;
        if (new_center_j < 0 || new_center_j >= j_max)
            continue;
        it->draw_new_star(img_old,img_temp,new_center_i,new_center_j);
        //做平滑处理
        //blur(img_temp, img_temp,Size(3,3));
        //img = img + img_temp;
        //img_temp.copyTo(img);
        //addWeighted(img, 1, img_temp, 1, 0, img); 
    }
}

class Mystars{
public:
    vector<Mystar*>stars;
    void add_point(int i,int j){
        bool flag_added = false;
        bool flag_skip = false;
        for(auto it :stars){
            if(it->try_to_add(i, j,flag_skip)){
                flag_added = true;
                break;
            }
        }
        if (flag_skip)
            return;
        if(!flag_added){
            stars.emplace_back(new Mystar(i,j));
        }
    }

    void show_stars(Mat &img_old){
        Mat img = img_old.clone();
        CvFont font;
	    cvInitFont(&font,CV_FONT_HERSHEY_SIMPLEX,1.0,1.0,0,2,8);
        for(auto it:stars){
            circle(img,Point(it->center_j,it->center_i),it->r+5,it->ave_color);
            putText(img, to_string(it->nums), Point(it->center_j,it->center_i), cv::FONT_HERSHEY_COMPLEX, 0.5, cv::Scalar(0, 255, 255), 1, 8, 0);
        }
        imshow("image", img);
        waitKey();
        imwrite("star.jpg", img);
    }
    
    void end_add(Mat &img){
        for(auto it :stars){
            it->calculate_r();
            it->calculate_color(img);
        }
    }

    void roll(Mat &img,double angle,int roll_i,int roll_j){
        double angle_abs = angle;
        double angle_step_used = angle_step;
        if (angle < 0){
            angle_abs = -angle;
            angle_step_used = -angle_step;
        }
        int i_max = img.rows;
        int j_max = img.cols;
        Mat img_old = img.clone();
        Mat img_temp = Mat::zeros(img.size(),img.type());

        boost::asio::thread_pool pool(30);
        for(auto it:stars){
            boost::asio::post(pool, boost::bind(roll_one_star, it,roll_i,roll_j,angle_abs,angle_step_used,boost::ref(img),boost::ref(img_old),boost::ref((img_temp))));
        }
        pool.join();
        blur(img_temp, img_temp,Size(3,3));
        //img = img + img_temp;
        //img_temp.copyTo(img);
        addWeighted(img, 1, img_temp, 1, 0, img); 

    }


};



int main(int argc, char* argv[])
{
    unsigned seed;  // Random generator seed
    // Use the time function to get a "seed” value for srand
    seed = time(0);
    srand(seed);

    const char* imagename = "./mengban.png";
    //从文件中读入图像
    Mat img = imread(imagename);
    Mat img_original = imread("./test2.jpg");
    //如果读入图像失败
    if(img.empty())
    {
        fprintf(stderr, "Can not load image %s\n", imagename);
        return -1;
    }
    int i_max = img.rows;
    int j_max = img.cols;
    cout << i_max<<endl;
    cout << j_max<<endl;
    Mystars stars;

    for(int i=0;i<i_max;++i){
        for (int j = 0; j < j_max; ++j) {
            auto &color = img.at<Vec3b>(i, j);//bgr
            if(color[0]>blue_threshold && color[1]>green_threshold && color[2]>red_threshold){
                stars.add_point(i, j);
            }
        }
    }

    stars.end_add(img);
    stars.show_stars(img);
    int roll_i = 600, roll_j = 400;
    stars.roll(img_original, -pi/9 , roll_i, roll_j);
    //circle(img, Point(roll_j, roll_i), 10, Vec3b{0,0,255});
    
    imshow("image", img_original);
    waitKey();
    imwrite("star roll.jpg", img_original);
    return 0;
}