a = 1.0;
b = 0.5;
learning_factor = 0.6;
k = -1;
neuron_selection_error = zeros(9,1);
mse_error = zeros(5000,1);

%define input and output set
x = [-0.7;0.3;-0.6;0.7;-0.5;0.9;1;-0.4;-0.3;0.1;0.5;-0.2;-0.1;
0;-1;-0.9;-0.8;0.6;0.2;0.4;0.8;];
actual_output = 0.8 * sin(pi*x);

for num_neurons = 2:10
    for fold = 1:3
        
        %division of test and training data for each of the samples
        if fold == 1
            train = x(1:14);
            train_output = actual_output(1:14);
            test = x(15:21);
            test_output = actual_output(15:21);
        end
        if fold == 2
            train = x(1:7);
            train(8:14,:) = x(15:21);
            test = x(8:14);
            test_output = actual_output(8:14);
        end
        if fold == 3
            train = x(8:21);
            train_output = actual_output(8:21);
            test = x(1:7);
            test_output = actual_output(1:7);
        end
        
        %initialization of weights with random values
        w = randi(100,2,num_neurons-1)/100;
        y = zeros(1,num_neurons);
        w_second = randi(100,1,num_neurons)/50;
        t = zeros(2, num_neurons-1);
        %iterations of training
        for i = 1:5000
            for j = 1:14
                z(1,1) = train(j);
                z(2,1) = -1;
                t(1,:) = w(1,:) * z(1);
                t(2,:) = w(2,:) * z(2);
                d = train_output(j);
                
                %forward phase - to 1st hidden layer
                w_sum = sum(t,1);
                for k = 1:num_neurons-1
                    y(k) = a*(-1+2/(1+exp(-b*w_sum(k))));
                end
                
                y(num_neurons) = -1;
                
                %forward phase - from hidden layer to output layer
                u = sum(y*w_second',2);
                output = a*(-1+2/(1+exp(-b*u)));
                desired_output = 0.8*sin(3.14*train(j));
                
                %calculation of f_dash
                f_dash = a*b*(1-output*output/a*a)/2;
                f_dash_first = zeros(num_neurons,1);
                for k = 1:num_neurons
                    f_dash_first(k) = a*b*(1-y(k)*y(k)/a*a)/2;
                end
                f_dash_first(num_neurons) = 0;
                
                %back propagation from output layer to hidden layer
                error = 0.5*(d-output)*(d-output);
                delta_op = (d-output) * f_dash;
                change_w_second = (learning_factor * delta_op) * y;
               
                
                w_second = w_second + change_w_second;
                %back propagation from hidden layer to input layer
                delta_y = (f_dash_first .* (w_second'*delta_op));
                delta_y = delta_y(1:num_neurons-1,:);
                change_prev_layer = learning_factor * (delta_y * z');
                
                %updating the weights of both the layers
                w = w + change_prev_layer';
            end
            output = zeros(21,1);
            
            %calculating validation error
            for j = 1:7
                t(1,:) = w(1,:) * test(j);
                t(2,:) = w(2,:) * -1;
                d = test_output(j);
                
                %forward phase - to 1st hidden layer
                w_sum = sum(t,1);
                for k = 1:num_neurons-1
                    y(k) = a*(-1+2/(1+exp(-b*w_sum(k))));
                end
                y(num_neurons) = -1;
                
                %forward phase - from hidden layer to output layer
                u = sum(y.*w_second,2);
                output(j) = a*(-1+2/(1+exp(-b*u)));
                
                mse_error(i) = mse_error(i) + ((d-output(j))^2);
            end
%             if i>1 && mse_error(i) > mse_error(i-1)
%                 break;
%             end
        end
    end
    mse_error = mse_error/3;
%     figure;
%     plot(mse_error);
    neuron_selection_error(num_neurons-1,1) = min(mse_error);
    
end
x = zeros(9,1);

for i = 2:10
    x(i-1) = i;
end

figure;
plot(x,neuron_selection_error);