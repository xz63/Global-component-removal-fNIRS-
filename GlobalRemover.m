classdef GlobalRemover
    % GlobalRemover
    % ---------------------------------------------------------------------
    % Author: Xian Zhang, PhD (Dr. Zhang, Xian)
    % Affiliation: Brain Function Lab, Department of Psychiatry,
    %              Yale School of Medicine
    % Contact: xzhang63@gmail.com
    %
    % Description:
    %   Implements spatial kernel-weighted estimation and removal of a global
    %   component from fNIRS time series based on angular distances
    %   between channel positions on the scalp. Channel neighborhoods are
    %   formed using a Gaussian kernel in spherical (TH, PHI) space.
    %
    % References (please cite when using this code):
    %   - "Separation of the global and local components in functional
    %      near-infrared spectroscopy signals using principal component
    %      spatial filtering"
    %   - "Signal processing of functional NIRS data acquired during overt
    %      speaking" (demonstrates global component in deoxy-Hb)
    %
    % Usage:
    %   gr = GlobalRemover(polhemus, sigmaDegrees, badChIdx);
    %   vGlobal = gr.getGlobal(vraw);
    %   vClean  = gr.remove(vraw);
    %
    % Inputs:
    %   xyz     (MNI coordinates Nx3 channel coords)
    %   sigmaDegrees  - Gaussian kernel width in degrees (scalar)
    % Notes:
    %   - Requires Statistics and Machine Learning Toolbox for pdist/squareform.
    %   - Coordinates in polhemus.NFRI_result.OtherC are assumed Cartesian (x,y,z).
    %   - Kernel weights are column-normalized to sum to 1.
    properties
        % Pairwise angular distance matrix (radians) computed from (TH, PHI)
        distance_matrix
        % Gaussian kernel width (degrees)
        sigma
        % Column-normalized spatial kernel (nCh x nCh)
        kernel
        xyz
        % Number of valid channels
        nCh
    end % end of propertieso
    methods(Static)
        function demo()
            load testdata
            xyz = meanxyz;
            sigma = 46;
            size(meandata) % ndata,nch, ncondition or nrun, 
            %we incourage contatinate all data because global compoment
            %should be shared with all condition and all runs
            nSignal=size(meandata,3); %2 oxy and dexoy
            data0=permute(meandata,[2 1 3]); data=data0(:,:); sz=size(data);
            badchannel=[ 1 ]; % a list of all bad channels
            badchannel=[  ]; % a list of all bad channels there is no bad channel
            ind=1:size(data,1); ind(badchannel)=[];
            gr = GlobalRemover(xyz(ind,:), sigma);
            globalC(ind,:)=gr.getGlobal(data(ind,:));
            cleanData(ind,:)=gr.remove(data(ind,:));
            cleanData=reshape(cleanData,[sz(1),sz(2)/nSignal,nSignal]);
            globalC=reshape(globalC,[sz(1),sz(2)/nSignal,nSignal]);

            figure; 
            subplot(2,2,1); imagesc(corrcoef(data0(:,:,1)'));caxis([-1 1]); title('data, Oxy');colorbar;
            subplot(2,2,2); imagesc(corrcoef(cleanData(:,:,1)')); caxis([-1 1]);  title('cleaned data, Oxy');colorbar;
            subplot(2,2,3); imagesc(corrcoef(globalC(:,:,1)')); caxis([-1 1]);  title('global component, Oxy');colorbar;
            subplot(2,2,4); imagesc(gr.kernel);   title('convolution kernel');colorbar;

            saveas(gcf,'globalremovaldemo.png');
            cleanData=permute(cleanData,[2,1,3]);
        end
    end
    methods
        function x = GlobalRemover(xyz, sigma)
            % Constructor
            %   Builds spatial kernel from channel coordinates while excluding
            %   specified bad channels.
            x.sigma = sigma;
            x.xyz = xyz;
            x.nCh = size(x.xyz, 1);
            % Convert to spherical to compute great-circle style angular distances
            [TH, PHI, ~] = cart2sph(x.xyz(:,1), x.xyz(:,2), x.xyz(:,3));
       
            for i=1:x.nCh
                for j=i:x.nCh
                    diffv(i,j,1)=TH(i)-TH(j);
                    diffv(i,j,2)=PHI(i)-PHI(j);
                    diffv(j,i,:)=diffv(i,j,:);
                end
            end
            for i=1:2
                temp=diffv(:,:,i);
                temp(temp>=pi)=temp(temp>=pi)-2*pi;    temp(temp<=-pi)=temp(temp<=-pi)+2*pi;
                diffv2(:,:,i)=temp;
            end
            % Pairwise angular distances via custom arclen metric
            x.distance_matrix =sqrt(diffv2(:,:,1).^2+diffv2(:,:,2).^2) ;
            % Gaussian kernel in angular space (sigma in degrees)
            a = (2 * (x.sigma * pi/180)^2);
            if x.sigma<20;
                warning('signa should be between 20-50 degrees')
            end

            kernel = exp(- x.distance_matrix.^2 / (2 * (x.sigma * pi/180)^2));

            % Column-normalize kernel so weights sum to 1 for each target channel
            x.kernel = zeros(size(kernel));
            for i = 1:x.nCh
                colSum = sum(kernel(:, i));
                if colSum == 0
                    x.kernel(:, i) = kernel(:, i);
                else
                    x.kernel(:, i) = kernel(:, i) / colSum;
                end
            end
        end

        function v = getGlobal(x, vraw)
            % Estimate the global component for each channel as a
            % kernel-weighted spatial average of signals across channels.
            v = (vraw' * x.kernel)';
        end

        function v = remove(x, vraw)
            % Remove the estimated global component from the raw signal.
            v = vraw - x.getGlobal(vraw);
        end
    end
end

