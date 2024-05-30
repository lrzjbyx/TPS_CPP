#include <iostream>
#include <vector>
#include <utility>
#include <iomanip>
#include <algorithm>
#include <opencv2/opencv.hpp>

std::vector<std::array<double, 2>> build_output_control_points(int num_control_points, std::pair<float,float>margins){
    float margin_x = margins.first;
    float margin_y = margins.second;
    int num_ctrl_pts_per_side = num_control_points / 2;

    std::vector<double> ctrl_pts_x(num_ctrl_pts_per_side);
    for (int i = 0; i < num_ctrl_pts_per_side; ++i) {
        ctrl_pts_x[i] = margin_x + i * (1.0 - 2 * margin_x) / (num_ctrl_pts_per_side - 1);
    }

    std::vector<std::array<double, 2>> ctrl_pts_top(num_ctrl_pts_per_side);
    std::vector<std::array<double, 2>> ctrl_pts_bottom(num_ctrl_pts_per_side);

    for (int i = 0; i < num_ctrl_pts_per_side; ++i) {
        ctrl_pts_top[i] = {ctrl_pts_x[i], margin_y};
        ctrl_pts_bottom[i] = {ctrl_pts_x[i], 1.0 - margin_y};
    }

    std::vector<std::array<double, 2>> output_ctrl_pts_arr;
    output_ctrl_pts_arr.insert(output_ctrl_pts_arr.end(), ctrl_pts_top.begin(), ctrl_pts_top.end());
    output_ctrl_pts_arr.insert(output_ctrl_pts_arr.end(), ctrl_pts_bottom.begin(), ctrl_pts_bottom.end());

    return output_ctrl_pts_arr;

}

std::vector<std::vector<double>> compute_partial_repr(std::vector<std::array<double, 2>> input_points,std::vector<std::array<double, 2>> control_points){
    std::vector<std::vector<double>> repr_matrix(input_points.size(), std::vector<double>(control_points.size()));

    for (size_t i = 0; i < input_points.size(); ++i) {
        for (size_t j = 0; j < control_points.size(); ++j) {
            double dx = input_points[i][0] - control_points[j][0];
            double dy = input_points[i][1] - control_points[j][1];
            double dist_squared = dx * dx + dy * dy;
            if (dist_squared > 0) {
                double repr_value = 0.5 * dist_squared * std::log(dist_squared);
                repr_matrix[i][j] = repr_value;
            } else {
                repr_matrix[i][j] = 0.0; // Handling the case where distance is zero
            }
        }
    }

    return repr_matrix;
}

std::vector<std::vector<double>> fill_forward_kernel(std::vector<std::vector<double>> input_points,std::vector<std::array<double, 2>> control_points,std::vector<std::vector<double>> &repr_matrix){
    int N = repr_matrix.size();
    int M = repr_matrix[0].size();

    // 0...N-3,0...N-3
    for(int i = 0; i < N-3; i++){
        for(int j = 0; j < M-3; j++){
            repr_matrix[i][j] = input_points[i][j];
        }
    }
    // 左侧 1
    for(int i = 0;i<N-3;i++){
        repr_matrix[i][N-3] = 1.0f;
    }
    // 右侧 1
    for(int i = 0;i<N-3;i++){
        repr_matrix[N-3][i] = 1.0f;
    }
    // 左侧
    for(int i = 0;i<N-3; i++){
        repr_matrix[i][N-2] = control_points[i][0];
        repr_matrix[i][N-1] = control_points[i][1];
    }
    // 下侧
    for(int i = 0;i<N-3; i++){
        repr_matrix[N-2][i] = control_points[i][0];
        repr_matrix[N-1][i] = control_points[i][1];
    }

    return repr_matrix;
}

bool invert_matrix(const std::vector<std::vector<double>>& input, std::vector<std::vector<double>>& inverse) {
    int n = input.size();
    std::vector<std::vector<double>> a(n, std::vector<double>(2 * n));

    // Create the augmented matrix [input | I]
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            a[i][j] = input[i][j];
        }
        a[i][n + i] = 1;
    }

    // Perform row operations
    for (int i = 0; i < n; i++) {
        // Zeroing elements below the diagonal
        if (a[i][i] == 0) { // Pivot 0 means the matrix may be singular
            bool singular = true;
            for (int k = i + 1; k < n; k++) {
                if (a[k][i] != 0) {
                    std::swap(a[i], a[k]);
                    singular = false;
                    break;
                }
            }
            if (singular) return false; // Matrix is singular
        }

        // Make the pivot element 1
        double d = a[i][i];
        for (int j = 0; j < 2 * n; j++) {
            a[i][j] /= d;
        }

        // Eliminate all other elements in the column
        for (int k = 0; k < n; k++) {
            if (k != i) {
                double factor = a[k][i];
                for (int j = 0; j < 2 * n; j++) {
                    a[k][j] -= a[i][j] * factor;
                }
            }
        }
    }

    // Extract the inverse matrix from the augmented form
    inverse.resize(n, std::vector<double>(n));
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            inverse[i][j] = a[i][n + j];
        }
    }

    return true;
}

std::vector<std::array<double, 2>> generate_grid_coordinates(int height, int width) {
    std::vector<std::array<double, 2>> coordinates;
    for (int i = 0; i < height; ++i) {
        for (int j = 0; j < width; ++j) {
            coordinates.push_back({static_cast<double >(i), static_cast<double >(j)});
        }
    }
    return coordinates;
}

std::vector<std::vector<double>> init_coordinates(std::vector<std::array<double, 2>> target_coordinate,std::vector<std::array<double, 2>> target_control_points,int target_height,int target_width){
    std::vector<double> X(target_coordinate.size()), Y(target_coordinate.size());

    for (size_t i = 0; i < target_coordinate.size(); ++i) {
        Y[i] = target_coordinate[i][0] / double(target_height - 1); // Normalize Y
        X[i] = target_coordinate[i][1] / double(target_width - 1);  // Normalize X
    }

    for (size_t i = 0; i < target_coordinate.size(); ++i) {
        target_coordinate[i][0] = X[i];
        target_coordinate[i][1] = Y[i];
    }

    std::vector<std::vector<double>> target_coordinate_partial_repr = compute_partial_repr(target_coordinate, target_control_points);
    std::vector<std::vector<double>> target_coordinate_repr(target_coordinate.size(), std::vector<double>(1 + target_coordinate_partial_repr[0].size() + 2));


    for (size_t i = 0; i < target_coordinate.size(); ++i) {
        size_t col = 0;
        // Add partial representations
        for (size_t j = 0; j < target_coordinate_partial_repr[i].size(); ++j, ++col) {
            target_coordinate_repr[i][col] = target_coordinate_partial_repr[i][j];
        }
        // Add a column of ones
        target_coordinate_repr[i][col++] = 1.0;
        // Add original coordinates
        target_coordinate_repr[i][col++] = target_coordinate[i][0];
        target_coordinate_repr[i][col++] = target_coordinate[i][1];
    }

    return target_coordinate_repr;

}

std::vector<std::vector<double>> matmul(const std::vector<std::vector<double>>& A, const std::vector<std::vector<double>>& B) {
    int rowsA = A.size();
    int colsA = A[0].size();
    int rowsB = B.size();
    int colsB = B[0].size();

    // Ensure matrix dimensions are compatible for multiplication
    if (colsA != rowsB) {
        throw std::invalid_argument("Matrix dimensions are not compatible for multiplication.");
    }

    std::vector<std::vector<double>> C(rowsA, std::vector<double>(colsB, 0.0));

    for (int i = 0; i < rowsA; ++i) {
        for (int j = 0; j < colsB; ++j) {
            for (int k = 0; k < colsA; ++k) {
                C[i][j] += A[i][k] * B[k][j];
            }
        }
    }

    return C;
}


std::vector<std::vector<std::array<double,2>>> compute_grid(std::vector<std::vector<double>> source_control_points,std::vector<std::vector<double>> target_coordinate_repr,std::vector<std::vector<double>> inverse_kernel,int target_height, int target_width){
    std::vector<std::vector<double>> padding_matrix = std::vector<std::vector<double>>(3, std::vector<double>(2, 0.0));
    std::vector<std::vector<double>> Y(source_control_points);
    Y.insert(Y.end(), padding_matrix.begin(), padding_matrix.end());

//    std::vector<std::vector<double>> inverse_kernel = {{1, 0}, {0, 1}}; // Identity matrix for simplicity
    std::vector<std::vector<double>> mapping_matrix = matmul(inverse_kernel, Y);

    // Example target coordinate representation (dummy data)
    std::vector<std::vector<double>> source_coordinate = matmul(target_coordinate_repr, mapping_matrix);

    // Reshape and clip
    std::vector<std::vector<std::array<double,2>>> grid(target_height, std::vector<std::array<double,2>>(target_width)); // Assuming reshaping works correctly

    for (int i = 0; i < target_height; ++i) {
        for (int j = 0; j < target_width; ++j) {
            const std::vector<double>& coord =  source_coordinate[i * target_width + j];
            grid[i][j][0] = std::clamp(coord[0] * 2.0 - 1.0, -1.0, 1.0);
            grid[i][j][1] = std::clamp(coord[1] * 2.0 - 1.0, -1.0, 1.0);
        }
    }



    return grid;

}

cv::Mat grid_sample(const cv::Mat& input, std::vector<std::vector<std::array<double, 2>>>& grid) {
    // 获取输入图像的维度
    int H = input.rows;
    int W = input.cols;
    int C = input.channels(); // Not used explicitly here, but useful for more complex operations

    int NH = grid.size();
    int NW = grid[0].size();


    // 创建映射矩阵
    cv::Mat map_x(NH, NW, CV_32F);
    cv::Mat map_y(NH, NW, CV_32F);

    // 调整网格坐标并填充映射矩阵
    for (int i = 0; i < NH; ++i) {
        for (int j = 0; j < NW; ++j) {
            auto& g = grid[i][j];  // g is std::array<double, 2>
            // Normalizing and scaling grid coordinates to match image dimensions
            double x = (g[0] + 1) * 0.5 * (W - 1);
            double y = (g[1] + 1) * 0.5 * (H - 1);

            map_x.at<float>(i, j) = static_cast<float>(x);
            map_y.at<float>(i, j) = static_cast<float>(y);
        }
    }

    // 创建输出图像
    cv::Mat output_image;
    cv::remap(input, output_image, map_x, map_y, cv::INTER_LINEAR);


    return output_image;
}

int main() {

    int num_control_points = 20;
    int target_height = 100;
    int target_width = 600;
    int M = num_control_points+3;
    std::pair<float,float> margins = std::make_pair(0.05,0.05);


    std::vector<std::vector<double>> source_control_points = {{0.301,0.273},
            {0.311, 0.195},
            {0.345, 0.124},
            {0.399, 0.066},
            {0.469, 0.026},
            {0.547, 0.007},
            {0.628, 0.011},
            {0.704, 0.038},
            {0.768, 0.085},
            {0.816, 0.149},
            {0.396, 0.271},
            {0.403, 0.221},
            {0.425, 0.174},
            {0.46 , 0.136},
            {0.505, 0.11 },
            {0.556, 0.098},
            {0.609, 0.101},
            {0.658, 0.118},
            {0.7  , 0.149},
            {0.731, 0.19 }};
    /**
     * 生成控制点
     * 本案例中控制点是自动生成的，也可通过指定控制点来进行变换，格式参照source_control_points
     * */
    auto output_ctrl_pts_arr = build_output_control_points(20,margins);
    /**
     * 计算输入点（input_points）和控制点（control_points）之间的一种欧几里得距离的平方
     * */
    std::vector<std::vector<double>> matrix = compute_partial_repr(output_ctrl_pts_arr, output_ctrl_pts_arr);

    /**
     * 依据公式对forward_kernel进行填充
     * */
    std::vector<std::vector<double>> forward_kernel(M, std::vector<double>(M));
    fill_forward_kernel(matrix,output_ctrl_pts_arr,forward_kernel);

    /**
     * 计算forward_kernel矩阵的逆
     * */
    std::vector<std::vector<double>> inverse_kernel(M, std::vector<double>(M));
    invert_matrix(forward_kernel,inverse_kernel);

    /**
     * 生成坐标表格
     * */
    std::vector<std::array<double, 2>> target_coordinate = generate_grid_coordinates(target_height, target_width);

    /**
     * 进行目标坐标的处理和转换
     * */
    std::vector<std::vector<double>> target_coordinate_repr = init_coordinates(target_coordinate,output_ctrl_pts_arr,target_height,target_width);
    /**
     * 计算映射矩阵
     * */
    std::vector<std::vector<std::array<double,2>>> grid = compute_grid(source_control_points,target_coordinate_repr,inverse_kernel,target_height, target_width);

    cv::Mat image = cv::imread("/Users/ruizhuti/Documents/code/cpp/clion/net/TPS_CPP/image.jpg");
    /**
     * 图片采样并变换
     * */
    cv::Mat output_image = grid_sample(image,grid);

    cv::imwrite("output_image.png", output_image);

    return 0;
}
