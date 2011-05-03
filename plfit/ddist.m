function ps=ddist()

    %type = 'test';
%    type = 'cooccurrence';
    type = 'dependency';

    %dataset = 'pl-test';
    %dataset = 'pl-test2';
    %dataset = 'tasa-giant';
    %dataset = 'air-giant';
%    dataset = 'tasa-docs';
    dataset = 'air-docs';

    num = 20;
    ps = dtestdocs(type, dataset, num);

%    [alpha, xmin, l] = dfit('cooccurrence', 'tasa-giant', 'xmax', 1000)
%    [alpha, xmin, l] = dfit('cooccurrence', 'air-giant', 'xmax', 1806)
%    [alpha, xmin, l] = dfit('dependency', 'tasa-giant', 'xmax', 596)
%    [alpha, xmin, l] = dfit('dependency', 'air-giant', 'xmax', 962)

%    [p, gof] = dtest('cooccurrence', 'tasa-giant', 1000)
%    [p, gof] = dtest('cooccurrence', 'air-giant', 1806)
%    [p, gof] = dtest('dependency', 'tasa-giant', 596)
%    [p, gof] = dtest('dependency', 'air-giant', 962)
    
%    [p, gof] = dtest('cooccurrence', 'tasa-giant', 30, 'xmax', 1000)
%    [p, gof] = dtest('cooccurrence', 'air-giant', 14, 'xmax', 1806)
%    [p, gof] = dtest('dependency', 'tasa-giant', 6, 'xmax', 596)
%    [p, gof] = dtest('dependency', 'air-giant', 6, 'xmax', 962)
    
%    [l, h] = dsplit('cooccurrence', 'tasa-giant', 1000)
%    [l, h] = dsplit('cooccurrence', 'air-giant', 1806)
%    [l, h] = dsplit('dependency', 'tasa-giant', 596)
%    [l, h] = dsplit('dependency', 'air-giant', 962)

%    dplot('dependency', 'tasa-giant', 3.58, 596, 'tail')
%    dplot('dependency', 'tasa-giant', 1.72, 6, 'base')
%    dplot('dependency', 'air-giant', 3.68, 962, 'tail')
%    dplot('dependency', 'air-giant', 1.59, 6, 'base')
%    dplot('cooccurrence', 'tasa-giant', 3.34, 1000, 'tail')
%    dplot('cooccurrence', 'tasa-giant', 1.82, 30, 'base')
%    dplot('cooccurrence', 'air-giant', 4.12, 1806, 'tail')
%    dplot('cooccurrence', 'air-giant', 1.62, 14, 'base')

function ps=dtestdocs(type, dataset, num)
    path = strcat('../output/degrees/',type,'/',dataset)
    data = dlmread(path);
    ps = zeros(1,num);
    for i = 1:num,
        i = i
        s = size(data)
        d = data(100+i,:);
        [alpha, xmin, l] = plfit(d,'range',[1.50:0.01:4.20]);
        %[h, fig] = plplot(data, xmin, alpha);
        [p,gof] = plpva(d,xmin)
        ps(1,i) = p;
    end

function dplot(type, dataset, alpha, xmin, postfix)
    path = strcat('../output/degrees/',type,'/',dataset)
    data = dlmread(path);
    [h, fig] = plplot(data, xmin, alpha);
    figfile = strcat('output/',type,'-',dataset,'-',postfix,'.png');
    print(fig, figfile, '-S640,480', '-dpng')

function [lower, higher]=dsplit(type, dataset, split)
    path = strcat('../output/degrees/',type,'/',dataset)
    data = dlmread(path);
    lower = length(data(data<split))
    higher = length(data(data>split))

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
    
    data = data(data>xmin);
    if limxmax, 
        data = data(data<xmax);
    end
    
    n_samples = length(data)
    
    [p, gof] = plpva(data, xmin)

