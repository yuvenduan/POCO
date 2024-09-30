import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
from configs.config_global import FIG_DIR

def adjust_figure(ax=None):
    if ax is None:
        ax = plt.gca()
    # Hide the right and top spines
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    plt.tight_layout(pad=0.5)

def heatmap_plot(
        data, 
        title=None, xlabel=None, ylabel=None, 
        cmap='viridis', fontsize=12, figsize=(5, 5),
        save_dir=None, fig_name=None):
    
    plt.figure(figsize=figsize)
    plt.imshow(data, cmap=cmap)
    plt.colorbar()

    if title:
        plt.title(title, fontsize=fontsize)
    if xlabel:
        plt.xlabel(xlabel, fontsize=fontsize)
    if ylabel:
        plt.ylabel(ylabel, fontsize=fontsize)

    if save_dir and fig_name:
        plt.savefig(os.path.join(save_dir, fig_name + '.pdf'))
    else:
        plt.show()
    plt.close()

def get_sem(data):
    return np.std(data, ddof=1) / np.sqrt(len(data))

def remove_nan(data):
    if isinstance(data, pd.Series):
        return data.dropna().values
    elif isinstance(data[0], (list, tuple, np.ndarray)):
        return [remove_nan(x) for x in data]
    return [val for val in data if val != None and not np.isnan(val)]

def error_plot(
    x_axis, 
    data_list,
    x_label=None,
    y_label=None,
    title=None,
    label_list=[None, ],
    save_dir=None,
    fig_name=None,
    legend_title=None,
    xticks=None,
    yticks=None,
    xticks_labels=None,
    yticks_labels=None,
    linewidth=2,
    capsize=5,
    capthick=1.5,
    fontsize=12,
    figsize=(4, 3),
    legend=True,
    legend_frameon=False,
    legend_loc='best',
    legend_fontsize=None,
    legend_bbox_to_anchor=None,
    extra_lines=None,
    colors=None,
    mode='errorbar1',
    errormode='sem',
    alphas=None,
    y_offsets=None,
    xlim=None,
    ylim=None,
    suffix='.pdf'
):
    """
    A generic error bar plot
    :param x_axis: A list representing the x-axis, or a list of lists representing x-axis for each curve
    :param data_list: A list of curves to plot,
        of size (num_curves, num_points, num_seeds)
        Each curve is a list of the same length as x_axis,
        where each element in the list is a list of integers
        containing results of different random seeds
    :param label_list: the corresponding labels for the curves
    :param extra_lines: a function that allows additional
        curves to be plotted, it should be defined by something like:
        def extra_lines(plt, x_axis, linewidth, capsize, capthick):
            plt.plot(...)
    :param mode: 'errorbar1', 'errorbar2', 'errorbar3', or 'errorshade'
    :param errormode: 'std' or 'sem'
    :param alphas: a list of alpha values for each curve
    :param y_offsets: a list of value to offset each curve
    """
    
    plt.figure(figsize=figsize)

    for i, data, label in zip(range(len(data_list)), data_list, label_list):

        data = remove_nan(data)
        mean = np.array([np.mean(val) for val in data])
        if y_offsets is not None:
            mean += y_offsets[i] * np.ones_like(mean)

        if errormode == 'std':
            error = np.array([np.std(val, ddof=1) for val in data])
        elif errormode == 'sem':
            error = np.array([get_sem(val) for val in data])
        elif errormode == 'none':
            error = np.zeros_like(mean)
        else:
            raise NotImplementedError('error mode not implemented')

        if colors is not None:
            color = colors[i]
        else:
            color = f'C{i}'

        if alphas is not None:
            alpha = alphas[i]
        else:
            alpha = 1.0

        if isinstance(x_axis[0], (tuple, list)):
            _x_axis = x_axis[i]
        else:
            _x_axis = x_axis

        if mode == 'errorbar1':
            # error bar plot with single color
            plt.errorbar(
                _x_axis,
                mean, error,
                label=label,
                linewidth=linewidth,
                capsize=capsize,
                capthick=capthick,
                fmt="o-",
                color=color,
                elinewidth=linewidth*0.6,
                markersize=linewidth*3.5,
                alpha=alpha
            )
        elif mode == 'errorbar2':
            # error bar plot with black outline
            plt.errorbar(
                _x_axis,
                mean, error,
                label=label,
                linewidth=linewidth,
                capsize=capsize,
                capthick=capthick,
                fmt="o-",
                mfc=color,
                ecolor='k',
                color='k',
                elinewidth=linewidth*0.6,
                markersize=linewidth*4.5,
                alpha=alpha
            )
        elif mode == 'errorbar3':
            # error bar plot without cap
            plt.errorbar(
                _x_axis,
                mean, error,
                label=label,
                linewidth=linewidth,
                fmt="o-",
                color=color,
                markeredgewidth=0,
                elinewidth=linewidth*0.6,
                markersize=linewidth*3.5,
                alpha=alpha
            )
        elif mode == 'errorshade':
            # show errors as shaded regions
            plt.plot(_x_axis, mean, label=label, linewidth=linewidth,
                     color=color, alpha=alpha)
            plt.fill_between(
                x=_x_axis,
                y1=mean - error,
                y2=mean + error,
                alpha=0.3*alpha,
                color=color,
                edgecolor=None,
            )
        else:
            raise NotImplementedError('Unknown mode')
        
    if extra_lines is not None:
        extra_lines(
            plt, x_axis, 
            linewidth=linewidth, 
            capsize=capsize, capthick=capthick
        )

    if x_label:
        plt.xlabel(x_label, fontsize=fontsize)
    if y_label:
        plt.ylabel(y_label, fontsize=fontsize)
    if title:
        plt.title(title, fontsize=fontsize)

    if xticks is not None:
        plt.xticks(xticks)
    plt.xticks(fontsize=fontsize, labels=xticks_labels)

    if yticks is not None:
        plt.yticks(yticks)
    plt.yticks(fontsize=fontsize, labels=yticks_labels)    

    if xlim is not None:
        plt.xlim(xlim)
    if ylim is not None:
        plt.ylim(ylim)    

    if legend and label_list != [None]:
        if legend_fontsize is None:
            legend_fontsize = fontsize
        plt.legend(title=legend_title, loc=legend_loc, frameon=legend_frameon,
                   fontsize=legend_fontsize, title_fontsize=legend_fontsize, bbox_to_anchor=legend_bbox_to_anchor)
    
    adjust_figure()

    save_dir = os.path.join(FIG_DIR, save_dir)
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(os.path.join(save_dir, f'{fig_name}{suffix}'))
    plt.close()

def errorbar_plot(
    data_list,
    label_list,
    position_list=None,
    color=None,
    **kwargs
):
    
    if position_list is None:
        position_list = range(len(data_list))

    x_axis = [[x] for x in position_list]
    data = [[y] for y in data_list]

    if isinstance(color, (list, tuple)):
        kwargs['colors'] = color
    elif color is not None:
        kwargs['colors'] = [color] * len(data_list)
    else:
        kwargs['colors'] = ['black'] * len(data_list)

    error_plot(
        x_axis, data, 
        xticks=position_list, xticks_labels=label_list, legend=None, 
        **kwargs
    )

def grouped_plot(
    data_list,
    group_labels,
    fig_size=(5, 3),
    fontsize=12,
    bar_labels=None,
    x_label=None,
    y_label=None,
    title=None,
    save_dir=None,
    fig_name=None,
    legend_title=None,
    legend=True,
    legend_frameon=False,
    legend_loc='best',
    legend_fontsize=None,
    legend_bbox_to_anchor=None,
    colors=None,
    bar_width=None,
    error_mode=None,
    style='violin',
    xlim=None,
    ylim=None,
    capsize=5,
    capthick=1.5,
    violin_alpha=0.5,
    show_bar_label=False
):
    """
    A grouped (bar) plot

    :param data_list: a list containing data for each atrribute, of size (num_bars, num_groups, (optional) num_seed)
    :param group_labels: labels for each group, would be shown on x-axis
    :param bar_labels: labels for each bar, would be shown on legend
    :param bar_width: width of each bar, default is 1 / (num_bars + 1)
    :param style: 'errorbar' or 'bar' or 'violin'
    """

    plt.figure(figsize=fig_size)

    if bar_width is None:
        bar_width = 1 / (len(data_list) + 1)

    if bar_labels is None:
        bar_labels = [None for data in data_list]
    
    for i, (attr, data) in enumerate(zip(bar_labels, data_list)):
        offset = (i - len(data_list) / 2 + 0.5) * bar_width
        x_axis = np.arange(len(data))
        
        if colors is not None:
            color = colors[i]
        else:
            color = f'C{i}'

        data = remove_nan(data)
        mean = np.array([np.mean(val) for val in data])

        if error_mode is not None:
            error = np.array([np.std(val, ddof=1) for val in data]) if error_mode == 'std' else np.array([get_sem(val) for val in data])
        else:
            error = None

        if style == 'violin':
            data = [np.array(val) for val in data]
            parts = plt.violinplot(data, x_axis + offset, widths=bar_width, showmeans=True)
            for pc in parts['bodies']:
                pc.set_facecolor(color)
                pc.set_alpha(violin_alpha)
            
            for idx, (x, y) in enumerate(zip(x_axis, data)):
                plt.scatter(
                    x + offset + np.linspace(-bar_width * 0.2, bar_width * 0.2, len(y)), y, 
                    color=color, s=12, marker='x',
                    label=attr if idx == 0 else None
                )

        elif style == 'errorbar':
            raise NotImplementedError('errorbar style is not supported for grouped plot')
            plt.errorbar(x_axis + offset, mean, yerr=error, label=attr, fmt='none', capsize=capsize, capthick=capthick, color=color)

        elif style == 'bar':
            rects = plt.bar(x_axis + offset, mean, bar_width, label=attr, color=color)
            if error_mode is not None:
                for x, y, yerr in zip(x_axis, mean, error):
                    plt.errorbar(x + offset, y, yerr=yerr, capsize=capsize, capthick=capthick, linewidth=capthick, color='black')
            if show_bar_label:
                plt.bar_label(rects, labels=[f'{val:.2f}' for val in mean], padding=3, fontsize=fontsize)
        else:
            raise NotImplementedError('Unknown style')

    
    plt.xticks(x_axis, group_labels, fontsize=fontsize)

    if x_label:
        plt.xlabel(x_label, fontsize=fontsize)
    if y_label:
        plt.ylabel(y_label, fontsize=fontsize)
    if title:
        plt.title(title, fontsize=fontsize)
    if xlim is not None:
        plt.xlim(xlim)
    if ylim is not None:
        plt.ylim(ylim)

    plt.xticks(fontsize=fontsize)
    plt.yticks(fontsize=fontsize)    

    if legend:
        if legend_fontsize is None:
            legend_fontsize = fontsize
        plt.legend(title=legend_title, loc=legend_loc, frameon=legend_frameon,
                   fontsize=legend_fontsize, title_fontsize=legend_fontsize, bbox_to_anchor=legend_bbox_to_anchor)
    
    adjust_figure()

    save_dir = os.path.join(FIG_DIR, save_dir)
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(os.path.join(save_dir, f'{fig_name}.pdf'))
    plt.close()

def histogram_plot(
    bins,
    data_list,
    x_label=None,
    y_label=None,
    title=None,
    label_list=[None, ],
    save_dir=None,
    fig_name=None,
    legend_title=None,
    xticks=None,
    yticks=None,
    fontsize=12,
    figsize=(6, 4),
    legend=True,
    legend_frameon=False,
    legend_loc='best',
    legend_fontsize=None,
    legend_bbox_to_anchor=None,
    extra_lines=None,
    colors=None,
    alphas=None,
    xlim=None,
    ylim=None,
    edgecolor='black'

):
    """
    Histogram plot for multiple distributions
    :param bins: A list representing the x-axis
    :param data_list: Each element in the list is a list of samples for a distribution
    :param label_list: the corresponding labels for the histograms
    :param extra_lines: a function that allows additional
        curves to be plotted, it should be defined by something like:
        def extra_lines(plt, x_axis):
            plt.plot(...)
    """
    
    plt.figure(figsize=figsize)

    for i, data, label in zip(range(len(data_list)), data_list, label_list):
        data = remove_nan(data)
        if colors is not None:
            color = colors[i]
        else:
            color = f'C{i}'

        if alphas is not None:
            alpha = alphas[i]
        else:
            alpha = 1.0

        plt.hist(data, bins, color=color, label=label, alpha=alpha, edgecolor=edgecolor)

    if x_label:
        plt.xlabel(x_label, fontsize=fontsize)
    if y_label:
        plt.ylabel(y_label, fontsize=fontsize)
    if title:
        plt.title(title, fontsize=fontsize)

    if xticks is not None:
        plt.xticks(xticks)
    plt.xticks(fontsize=fontsize)

    if yticks is not None:
        plt.yticks(yticks)
    plt.yticks(fontsize=fontsize)    

    if xlim is not None:
        plt.xlim(xlim)
    if ylim is not None:
        plt.ylim(ylim)

    if extra_lines is not None:
        extra_lines(plt, bins)

    if legend:
        if legend_fontsize is None:
            legend_fontsize = fontsize
        plt.legend(title=legend_title, loc=legend_loc, frameon=legend_frameon,
                   fontsize=legend_fontsize, title_fontsize=legend_fontsize, bbox_to_anchor=legend_bbox_to_anchor)
    
    adjust_figure()

    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(os.path.join(save_dir, f'{fig_name}.pdf'))
    plt.close()

def scatter_plot(
    X, Y, 
    colors, sizes=5,
    x_label=None,
    y_label=None,
    title=None,
    save_dir=None,
    fig_name=None,
    legend_title=None,
    xticks=None,
    yticks=None,
    fontsize=12,
    figsize=(4, 4),
    xlim=None,
    ylim=None,
    legend=False,
    legend_frameon=False,
    legend_loc='best',
    legend_fontsize=None,
    legend_bbox_to_anchor=None,
):
    
    plt.figure(figsize=figsize)
    plt.scatter(X, Y, c=colors, s=sizes)

    if x_label:
        plt.xlabel(x_label, fontsize=fontsize)
    if y_label:
        plt.ylabel(y_label, fontsize=fontsize)
    if title:
        plt.title(title, fontsize=fontsize)

    if xticks is not None:
        plt.xticks(xticks)
    plt.xticks(fontsize=fontsize)

    if yticks is not None:
        plt.yticks(yticks)
    plt.yticks(fontsize=fontsize)    

    if xlim is not None:
        plt.xlim(xlim)
    if ylim is not None:
        plt.ylim(ylim)

    if legend:
        raise NotImplementedError('legend is not yet supported for scatter plot')
        if legend_fontsize is None:
            legend_fontsize = fontsize
        plt.legend(title=legend_title, loc=legend_loc, frameon=legend_frameon,
                   fontsize=legend_fontsize, title_fontsize=legend_fontsize, bbox_to_anchor=legend_bbox_to_anchor)
    
    adjust_figure()

    save_dir = os.path.join(FIG_DIR, save_dir)
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(os.path.join(save_dir, f'{fig_name}.pdf'))
    plt.close()


def distribution_plot(
    data, 
    label_list, 
    left, right, interval, 
    plot_name,
    plot_dir,
    x_label='Value',
    y_label='Frequency',
    mode='errorshade',
    coef=1, 
    log=False,
    **kwargs
):
    """
    :param data: 3D list of shape (num_labels, num_seeds, num_trials),
        the last dimension is a list of scalar values
    """

    x_axis = list(np.arange(left, right + 0.01, interval))
    curve_list = []

    for datum in data:
        data_list = []
        
        for x in x_axis:
            densities = [((error_val > x - interval / 2) &
                          (error_val <= x + interval / 2)).mean() / interval * coef
                         for error_val in datum]
            data_list.append(densities)
        
        if log:
            print(data_list)
            data_list = np.log(data_list)
        curve_list.append(data_list)

    error_plot(
        x_axis,
        curve_list,
        x_label,
        y_label,
        label_list=label_list,
        save_dir=plot_dir,
        fig_name=plot_name,
        figsize=(3.5, 3.5),
        mode=mode,
        **kwargs
    )

def pred_vs_target_plot(
    data, 
    save_dir, 
    fig_name,
    vertical_line_pos=None,
    fontsize=8,
    legend=True
):
    """
    data: a list of tuples (pred, target), both are 1D numpy arrays of the same length; each tuple is a trial and would be plotted as in a seperate subplot
    """

    fig_size = (4, len(data) * 0.8 + 0.5)
    fig, axes = plt.subplots(len(data), 1, figsize=fig_size)

    ylim = max([max(np.max(np.abs(target)), np.max(np.abs(pred))) for pred, target in data])
    ylim = max(ylim, 1)

    for i, (prediction, target) in enumerate(data):
        ax = axes[i]
        ax.plot(target, label='target', color='gray')
        ax.plot(prediction, label='prediction', color='orange')
        # ax.set_ylim(-ylim, ylim)

        if i == 0 and legend:
            ax.legend(fontsize=fontsize)
        
        if i != len(data) - 1:
            ax.set_xticks([])

        if vertical_line_pos is not None:
            ax.axvline(vertical_line_pos, color='black', linestyle='--')

    save_dir = os.path.join(FIG_DIR, save_dir)
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(os.path.join(save_dir, f'{fig_name}.pdf'))
    plt.close()