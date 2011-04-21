function ddist()

    type = 'test';
    %type = 'cooccurrence';
    %type = 'dependency';

    dataset = 'pl-test';
    %dataset = 'pl-test2';
    %dataset = 'tasa-giant';
    %dataset = 'air-giant';
    %dataset = 'tasa-docs';
    %dataset = 'air-docs';

%    dtest(type, dataset, 3)        

%    dfit('cooccurrence', 'tasa-giant', 'xmax', 1000)
%    dfit('cooccurrence', 'air-giant', 'xmax', 1806)
%    dfit('dependency', 'tasa-giant', 'xmax', 596)
%    dfit('dependency', 'air-giant', 'xmax', 962)

%    dtest('cooccurrence', 'tasa-giant', 1000)
    dtest('cooccurrence', 'air-giant', 1806)
    %dtest('dependency', 'tasa-giant', 596)
    %dtest('dependency', 'air-giant', 962)
    
    %dtest('cooccurrence', 'tasa-giant', 30, 'xmax', 1000)
    %dtest('cooccurrence', 'air-giant', 14, 'xmax', 1806)
    %dtest('dependency', 'tasa-giant', 6, 'xmax', 596)
    %dtest('dependency', 'air-giant', 6, 'xmax', 962)

function [alpha, xmin, l]=dfit(type, dataset, varargin)
    path = strcat('../output/degrees/',type,'/',dataset)
    data = dlmread(path);
    
    limxmax = false;
    savefig = false;
    i = 1;
    while i<=length(varargin), 
        if ischar(varargin{i}), 
            switch varargin{i},
                case 'xmax',  limxmax = true; xmax=varargin{i+1};   i = i+1;
                case 'savefig', savefig = true; i = i+1;
            end
        end
        i = i + 1;
    end
    
    if limxmax, 
        data = data(data<xmax);
    end
    
    [alpha, xmin, l] = plfit(data,'range',[1.50:0.01:4.20])
    [h, fig] = plplot(data, xmin, alpha);

    if savefig,
        figfile = strcat('output/',type,'-',dataset,'-fitted.png');
        print(fig, figfile, '-S640,480', '-dpng')
    end
    
function [p, gof]=dtest(type, dataset, xmin, varargin)
    path = strcat('../output/degrees/',type,'/',dataset)
    data = dlmread(path);
    
    limxmax = false;
    i = 1;
    while i<=length(varargin), 
        if ischar(varargin{i}), 
            switch varargin{i},
                case 'xmax',  limxmax = true; xmax=varargin{i+1};   i = i+1;
            end
        end
        i = i + 1;
    end
    
    if limxmax, 
        data = data(data<xmax);
    end
    
    [p, gof] = plpva(data, xmin, 'reps', 1000)

